import argparse
import os
from copy import deepcopy
from subprocess import PIPE, STDOUT, Popen
from threading import Thread
from typing import Any, AsyncGenerator, Dict, List, Union

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from sse_starlette import EventSourceResponse

from ..utils.generic import dictify
from .common import create_stream_chunk, print_subprocess_stdout
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Embeddings,
    EmbeddingsRequest,
    EmbeddingsResponse,
    Finish,
    Function,
    FunctionCall,
    ModelCard,
    ModelList,
    Role,
    UsageInfo,
)


async def _stream_openai_chat_completion(response: AsyncStream[ChatCompletionChunk]) -> AsyncGenerator[str, None]:
    request_id, model = None, None
    async for chunk in response:
        if request_id is None:
            request_id = chunk.id
            model = chunk.model
            yield create_stream_chunk(request_id, model, ChatCompletionMessage(role=Role.ASSISTANT, content=""))

        new_token = chunk.choices[0].delta.content
        if new_token:
            yield create_stream_chunk(request_id, model, ChatCompletionMessage(content=new_token))

    yield create_stream_chunk(request_id, model, ChatCompletionMessage(), finish_reason=Finish.STOP)
    yield "[DONE]"


async def _create_openai_chat_completion(
    request: "ChatCompletionRequest", model: "AsyncOpenAI"
) -> "ChatCompletionResponse":
    response: "ChatCompletion" = await model.chat.completions.create(
        model=request.model,
        messages=[dictify(message) for message in request.messages],
        tools=[dictify(tool) for tool in request.tools] if request.tools is not None else None,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    if request.stream:
        return EventSourceResponse(_stream_openai_chat_completion(response), media_type="text/event-stream")

    choices = []
    for i, choice in enumerate(response.choices):
        if choice.message.tool_calls is not None:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_calls.append(FunctionCall(id=tool_call.id, function=Function(name=name, arguments=arguments)))

            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=tool_calls),
                    finish_reason=Finish.TOOL,
                )
            )
        else:
            content = choice.message.content
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatCompletionMessage(role=Role.ASSISTANT, content=content),
                    finish_reason=Finish.STOP,
                )
            )

    usage = UsageInfo(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
    return ChatCompletionResponse(
        id=response.id,
        model=response.model,
        choices=choices,
        usage=usage,
    )


async def _create_openai_embeddings(request: "EmbeddingsRequest", model: "AsyncOpenAI") -> "EmbeddingsResponse":
    response = await model.embeddings.create(
        model=request.model,
        input=request.input,
        encoding_format=request.encoding_format,
    )

    embeddings = []
    for i, embedding in enumerate(response.data):
        embeddings.append(Embeddings(embedding=embedding.embedding, index=i))

    usage = UsageInfo(
        prompt_tokens=response.usage.prompt_tokens,
        total_tokens=response.usage.total_tokens,
    )
    return EmbeddingsResponse(
        data=embeddings,
        model=response.model,
        usage=usage,
    )


def _launch_chat_server(chat_config: Dict[str, Any]) -> "Popen":
    cmd = "python -m imitater.service.chat"
    cmd += " --name {}".format(chat_config["name"])
    cmd += " --path {}".format(chat_config["path"])
    cmd += " --device {}".format(" ".join(map(str, chat_config["device"])))
    cmd += " --port {}".format(chat_config["port"])
    cmd += " --maxlen {}".format(chat_config["maxlen"]) if "maxlen" in chat_config else ""
    cmd += " --agent_type {}".format(chat_config["agent_type"]) if "agent_type" in chat_config else ""
    cmd += " --template {}".format(chat_config["template"]) if "template" in chat_config else ""
    cmd += " --gen_config {}".format(chat_config["gen_config"]) if "gen_config" in chat_config else ""
    env = deepcopy(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, chat_config["device"]))
    return Popen(cmd, env=env, shell=True, stdout=PIPE, stderr=STDOUT)


def _launch_embed_server(embed_config: Dict[str, Any]) -> "Popen":
    cmd = "python -m imitater.service.embed"
    cmd += " --name {}".format(embed_config["name"])
    cmd += " --path {}".format(embed_config["path"])
    cmd += " --device {}".format(" ".join(map(str, embed_config["device"])))
    cmd += " --port {}".format(embed_config["port"])
    cmd += " --batch_size {}".format(embed_config["batch_size"]) if "batch_size" in embed_config else ""
    env = deepcopy(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, embed_config["device"]))
    return Popen(cmd, env=env, shell=True, stdout=PIPE, stderr=STDOUT)


def launch_server(config_file: str) -> None:
    with open(config_file, "r", encoding="utf-8") as f:
        config: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = yaml.safe_load(f)

    port = config["service"].get("port", 8000)
    chat_models: Dict[str, "AsyncOpenAI"] = {}
    embed_models: Dict[str, "AsyncOpenAI"] = {}
    processes: List["Popen"] = []

    for chat_config in config["chat"]:
        if "token" in chat_config:
            chat_models[chat_config["name"]] = AsyncOpenAI(api_key=chat_config["token"])
        else:
            processes.append(_launch_chat_server(chat_config))
            chat_models[chat_config["name"]] = AsyncOpenAI(
                api_key="0", base_url="http://localhost:{}/v1".format(chat_config["port"])
            )

    for embed_config in config["embed"]:
        if "token" in embed_config:
            embed_models[embed_config["name"]] = AsyncOpenAI(api_key=embed_config["token"])
        else:
            processes.append(_launch_embed_server(embed_config))
            embed_models[embed_config["name"]] = AsyncOpenAI(
                api_key="0", base_url="http://localhost:{}/v1".format(embed_config["port"])
            )

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_names = set()
        model_names.update(chat_models.keys())
        model_names.update(embed_models.keys())
        return ModelList(data=[ModelCard(id=name) for name in model_names])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: "ChatCompletionRequest"):
        if request.model not in chat_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
        return await _create_openai_chat_completion(request, chat_models[request.model])

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
    async def create_embeddings(request: "EmbeddingsRequest"):
        if request.model not in embed_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
        return await _create_openai_embeddings(request, embed_models[request.model])

    for process in processes:
        thread = Thread(target=print_subprocess_stdout, args=[process])
        thread.start()

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    launch_server(getattr(args, "config"))
