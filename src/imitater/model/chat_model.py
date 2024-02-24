from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, GenerationConfig
from typing_extensions import Self
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from ..agent import get_agent, list_agents
from ..utils.modelscope import try_download_model_from_ms


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


@dataclass
class ChatConfig:
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
        parser.add_argument("--device", type=int, nargs="+")
        parser.add_argument("--port", type=int)
        parser.add_argument("--maxlen", type=int, default=2048)
        parser.add_argument("--agent_type", type=str, choices=list_agents(), default="react")
        parser.add_argument("--template", type=str, default=None)
        parser.add_argument("--gen_config", type=str, default=None)

    @classmethod
    def from_cli_args(cls, args: "Namespace") -> Self:
        attrs = [attr.name for attr in fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class ChatModel:
    r"""
    Creates a chat model for chat completions.

    Methods:
        chat: generates chat completions.
        stream_chat: streams chat completions.
        function_call: generates tool calls.
    """

    def __init__(self, config: "ChatConfig") -> None:
        config.path = try_download_model_from_ms(config.path)
        self.config = config
        self.name = config.name
        self._agent = get_agent(config.agent_type)
        self._init_vllm_engine()
        self._load_tokenizer()
        self._load_generation_config()

    def _init_vllm_engine(self) -> None:
        engine_args = AsyncEngineArgs(
            model=self.config.path,
            trust_remote_code=True,
            max_model_len=self.config.maxlen,
            tensor_parallel_size=len(self.config.device),
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

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

        self._engine.engine.tokenizer.tokenizer.add_special_tokens(
            {"additional_special_tokens": extra_special_tokens}, replace_additional_special_tokens=False
        )

    async def _generate(self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs):
        input_ids = self._tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=gen_kwargs.pop("temperature", None) or self._generation_config.temperature,
            top_p=gen_kwargs.pop("top_p", None) or self._generation_config.top_p,
            max_tokens=gen_kwargs.pop("max_tokens", None) or self._generation_config.max_new_tokens,
            stop_token_ids=self._generation_config.eos_token_id + gen_kwargs.pop("stop_token_ids", []),
        )
        result_generator = self._engine.generate(
            prompt=None, sampling_params=sampling_params, request_id=request_id, prompt_token_ids=input_ids
        )
        return result_generator

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
        generator = await self._generate(messages, request_id, **gen_kwargs)
        async for result in generator:
            if not result.finished:
                generated_text = result.outputs[0].text
                prompt_tokens = len(result.prompt_token_ids)
                completion_tokens = len(result.outputs[0].token_ids)

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
        generator = await self._generate(messages, request_id, **gen_kwargs)
        async for result in generator:
            if not result.finished:
                delta_text = result.outputs[0].text[len(generated_text) :]
                generated_text = result.outputs[0].text
                yield delta_text

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
