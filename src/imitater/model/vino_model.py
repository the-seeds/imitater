from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, GenerationConfig
from typing_extensions import Self

from imitater.agent import get_agent, list_agents



if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import OV_XML_FILE_NAME

from transformers import (PretrainedConfig, AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
from threading import Thread
import torch


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class OVCHATGLMModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for CHATGLM2
    """

    def __init__(
            self,
            model: "Model",
            config: "PretrainedConfig" = None,
            device: str = "CPU",
            dynamic_shapes: bool = True,
            ov_config: Optional[Dict[str, str]] = None,
            model_save_dir: Optional[Union[str, Path]] = None,
            **kwargs,
    ):
        NormalizedConfigManager._conf["chatglm"] = NormalizedTextConfig.with_args(
            num_layers="num_hidden_layers",
            num_attention_heads="num_attention_heads",
            hidden_size="hidden_size",
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )

    def _reshape(
            self,
            model: "Model",
            *args, **kwargs
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith('beam_idx'):
                continue
            if input_name.startswith('past_key_values'):
                shapes[inputs][1] = -1
                shapes[inputs][2] = 2
            elif shapes[inputs].rank.get_length() > 1:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
            cls,
            model_id: Union[str, Path],
            config: PretrainedConfig,
            use_auth_token: Optional[Union[bool, str, None]] = None,
            revision: Optional[Union[str, None]] = None,
            force_download: bool = False,
            cache_dir: Optional[str] = None,
            file_name: Optional[str] = None,
            subfolder: str = "",
            from_onnx: bool = False,
            local_files_only: bool = False,
            load_in_8bit: bool = False,
            **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path)
        init_cls = OVCHATGLMModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


@dataclass
class VinoChatConfig:
    r"""
    Creates configuration for a chat model.

    Methods:
        add_cli_args: adds arguments to a argument parser.
        from_cli_args: builds configuration based on the command line arguments.
    """

    name: str
    path: str
    device: List[int]
    port: int
    maxlen: int
    agent_type: str
    template: Optional[str]
    gen_config: Optional[str]

    @staticmethod
    def add_cli_args(parser: "ArgumentParser") -> None:
        parser.add_argument("--name", type=str)
        parser.add_argument("--path", type=str)
        parser.add_argument("--device", type=str)
        parser.add_argument("--port", type=int)
        parser.add_argument("--maxlen", type=int, default=2048)
        parser.add_argument("--agent_type", type=str, choices=list_agents(), default="react")
        parser.add_argument("--template", type=str, default=None)
        parser.add_argument("--gen_config", type=str, default=None)

    @classmethod
    def from_cli_args(cls, args: "Namespace") -> Self:
        attrs = [attr.name for attr in fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class VinoChatModel:
    r"""
    Creates a chat model for chat completions.

    Methods:
        chat: generates chat completions.
        stream_chat: streams chat completions.
        function_call: generates tool calls.
    """

    def __init__(self, config: "VinoChatConfig") -> None:
        self.config = config
        self.name = config.name
        self._agent = get_agent(config.agent_type)
        self._init_vino_engine()
        self._load_tokenizer()
        self._load_generation_config()

    def _init_vino_engine(self) -> None:
        ov_config = {"PERFORMANCE_HINT": "LATENCY",
                     "NUM_STREAMS": "1", "CACHE_DIR": ""}
        self._engine = OVCHATGLMModel.from_pretrained(
            self.config.path,
            device=self.config.device,
            ov_config=ov_config,
            config=AutoConfig.from_pretrained(self.config.path, trust_remote_code=True),
            trust_remote_code=True,
        )

    def _load_tokenizer(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
        if self.config.template:
            with open(self.config.template, "r", encoding="utf-8") as f:
                self._tokenizer.chat_template = f.read()

        if self._tokenizer.chat_template is None:
            print("Chat template is not found, use the default one.")

    def _load_generation_config(self) -> None:
        try:
            generation_config_path = self.config.gen_config or self.config.path
            self._generation_config = GenerationConfig.from_pretrained(generation_config_path)
        except Exception:
            self._generation_config = GenerationConfig(
                pad_token_id=self._tokenizer.pad_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        if not self._generation_config.temperature:
            self._generation_config.temperature = 1.0

        if not self._generation_config.top_p:
            self._generation_config.top_p = 1.0

        if not self._generation_config.max_new_tokens:
            self._generation_config.max_new_tokens = 2048

        if isinstance(self._generation_config.eos_token_id, int):
            self._generation_config.eos_token_id = [self._generation_config.eos_token_id]

        extra_special_tokens = []
        for eos_token_id in self._generation_config.eos_token_id:
            if eos_token_id != self._tokenizer.eos_token_id:
                extra_special_tokens.append(self._tokenizer.convert_ids_to_tokens(eos_token_id))

        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": extra_special_tokens}, replace_additional_special_tokens=False
        )

    async def _generate(self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs):
        input_ids = self._tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        )
        streamer = TextIteratorStreamer(
            self._tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True
        )
        stop_tokens = self._generation_config.eos_token_id + gen_kwargs.pop("stop_token_ids", [])
        stop_tokens = [StopOnTokens(stop_tokens)]

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self._generation_config.max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList(stop_tokens)
        )

        t1 = Thread(target= self._engine.generate, kwargs=generate_kwargs)
        t1.start()


        return streamer

    async def chat(self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs) -> Tuple[str, int, int]:
        r"""
        Generates chat completions.

        Args:
            messages: input messages.
            request_id: request ID.
            temperature: generation parameter.
            top_p: generation parameter.
            max_tokens: generation parameter.
            stop_token_ids: generation parameter.

        Returns:
            generated_text: the generated text.
            prompt_tokens: the number of prompt tokens.
            completion_tokens: the number of completion tokens.
        """
        generated_text, prompt_tokens, completion_tokens = "", 0, 0
        streamer = await self._generate(messages, request_id, **gen_kwargs)
        for new_text in streamer:
            new_text = new_text
            print(new_text, end="", flush=True)
            generated_text += new_text

        return generated_text, prompt_tokens, completion_tokens

    async def stream_chat(
            self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs
    ) -> AsyncGenerator[str, None]:
        r"""
        Streams chat completions.

        Args:
            messages: input messages.
            request_id: request ID.
            temperature: generation parameter.
            top_p: generation parameter.
            max_tokens: generation parameter.
            stop_token_ids: generation parameter.

        Returns:
            generated_token: the generated token.
        """
        generated_text = ""
        streamer = await self._generate(messages, request_id, **gen_kwargs)
        partial_text = ""
        for new_text in streamer:
            new_text = new_text
            print(new_text, end="", flush=True)
            partial_text += new_text
            yield new_text

    async def function_call(
            self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], request_id: str, **gen_kwargs
    ) -> Tuple[Union[str, Tuple[str, str]], int, int]:
        r"""
        Generates chat completions.

        Args:
            messages: input messages.
            tools: tools available.
            request_id: request ID.
            temperature: generation parameter.
            top_p: generation parameter.
            max_tokens: generation parameter.
            stop_token_ids: generation parameter.

        Returns:
            response | (name, arguments): response text or tool name with JSON arguments if tool exists.
            prompt_tokens: the number of prompt tokens.
            completion_tokens: the number of completion tokens.
        """
        agent_messages = self._agent.build_prompt(messages, tools)
        stop_word = self._agent.get_stop_word()
        if stop_word is not None:
            stop_word_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(stop_word)[0])
            gen_kwargs["stop_token_ids"] = gen_kwargs.pop("stop_token_ids", []) + [stop_word_id]

        generated_text, prompt_tokens, completion_tokens = await self.chat(agent_messages, request_id, **gen_kwargs)

        if stop_word is not None:
            stop_token = self._tokenizer.convert_ids_to_tokens(stop_word_id)
            if generated_text.endswith(stop_token):
                generated_text = generated_text[: -len(stop_token)]

        return self._agent.extract_tool(generated_text, tools), prompt_tokens, completion_tokens

