import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Generator, List, Tuple, Union

from transformers import AutoTokenizer, GenerationConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from ..agent import get_agent
from ..utils.vllm_monkey_patch import llama_attn_bias_monkey_patch


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import RequestOutput


class ChatModel:
    def __init__(self) -> None:
        self._agent = get_agent(os.environ.get("AGENT_TYPE"))
        self._init_vllm_engine()
        self._load_tokenizer()
        self._load_generation_config()

    def _init_vllm_engine(self) -> None:
        if os.environ.get("ENABLE_ATTN_BIAS") and int(os.environ.get("ENABLE_ATTN_BIAS")):
            llama_attn_bias_monkey_patch()

        engine_args = AsyncEngineArgs(model=os.environ.get("CHAT_MODEL_PATH"), trust_remote_code=True)
        if os.environ.get("CHAT_MODEL_DEVICE"):
            engine_args.tensor_parallel_size = len(os.environ.get("CHAT_MODEL_DEVICE").split(","))

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _load_tokenizer(self) -> None:
        self._tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.environ.get("CHAT_MODEL_PATH"), trust_remote_code=True
        )

        if os.environ.get("CHAT_TEMPLATE_PATH"):
            with open(str(os.environ.get("CHAT_TEMPLATE_PATH")), "r", encoding="utf-8") as f:
                self._tokenizer.chat_template = f.read()

        if self._tokenizer.chat_template is None:
            print("Chat template is not found, use the default one.")

    def _load_generation_config(self) -> None:
        try:
            generation_config_path = os.environ.get("GENERATION_CONFIG_PATH") or os.environ.get("CHAT_MODEL_PATH")
            self._generation_config = GenerationConfig.from_pretrained(pretrained_model_name=generation_config_path)
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
            self._generation_config.max_new_tokens = 1024

        if isinstance(self._generation_config.eos_token_id, int):
            self._generation_config.eos_token_id = [self._generation_config.eos_token_id]

    async def _generate(
        self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs
    ) -> AsyncIterator["RequestOutput"]:
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

    async def chat(self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs) -> str:
        generator = await self._generate(messages, request_id, **gen_kwargs)
        generated_text = ""
        async for result in generator:
            generated_text = result.outputs[0].text

        return generated_text

    async def stream_chat(
        self, messages: List[Dict[str, str]], request_id: str, **gen_kwargs
    ) -> Generator[str, None, None]:
        generator = await self._generate(messages, request_id, **gen_kwargs)
        generated_text = ""
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text) :]
            generated_text = result.outputs[0].text
            yield delta_text

    async def function_call(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], request_id: str, **gen_kwargs
    ) -> Union[str, Tuple[str, str]]:
        agent_messages = self._agent.build_prompt(messages, tools)
        stop_word = self._agent.get_stop_word()
        if stop_word is not None:
            gen_kwargs["stop_token_ids"] = [self._tokenizer.encode(stop_word)[0]]

        generator = await self._generate(agent_messages, request_id, **gen_kwargs)
        generated_text = ""
        async for result in generator:
            generated_text = result.outputs[0].text

        if stop_word is not None:
            stop_token = self._tokenizer.decode(gen_kwargs["stop_token_ids"])
            if generated_text.endswith(stop_token):
                generated_text = generated_text[: -len(stop_token)]

        return self._agent.extract_tool(generated_text, tools)
