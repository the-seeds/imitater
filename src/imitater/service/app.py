import argparse
import os
from subprocess import PIPE, STDOUT, Popen
from threading import Thread
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from sse_starlette import EventSourceResponse

from ..utils.generic import dictify, jsonify
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
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


def read_message(process: "Popen") -> None:
    while process.stdout.readable():
        line: bytes = process.stdout.readline()

        if not line:
            break

        print(line.decode("utf-8").strip())


def create_stream_chunk(
    request_id: str, model: str, delta: "ChatCompletionMessage", finish_reason: Optional[Finish] = None
) -> str:
    choice = ChatCompletionStreamResponseChoice(index=0, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=request_id, model=model, choices=[choice])
    return jsonify(chunk)


async def stream(response: AsyncStream[ChatCompletionChunk]) -> AsyncGenerator[str, None]:
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


async def create_chat_completion(
    request: "ChatCompletionRequest", chat_model: "AsyncOpenAI"
) -> "ChatCompletionResponse":
    response: "ChatCompletion" = await chat_model.chat.completions.create(
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
        return EventSourceResponse(stream(response), media_type="text/event-stream")

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


async def create_embeddings(request: "EmbeddingsRequest", embed_model: "AsyncOpenAI"):
    response = await embed_model.embeddings.create(
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


def launch_server(config_file: str) -> None:
    with open(config_file, "r", encoding="utf-8") as f:
        config: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = yaml.safe_load(f)

    port = config["service"].get("port", 8000)
    chat_models: Dict[str, "AsyncOpenAI"] = {}
    embed_models: Dict[str, "AsyncOpenAI"] = {}
    processes: List["Popen"] = []

    for chat_config in config["chat"]:
        cmd = "python -m imitater.service.chat"
        cmd += " --name {}".format(chat_config.get("name"))
        cmd += " --path {}".format(chat_config.get("path"))
        cmd += " --device {}".format(" ".join(map(str, chat_config.get("device"))))
        cmd += " --port {}".format(chat_config.get("port"))
        if chat_config.get("maxlen", None):
            cmd += " --maxlen {}".format(chat_config.get("maxlen"))
        if chat_config.get("agent_type", None):
            cmd += " --agent_type {}".format(chat_config.get("agent_type"))
        if chat_config.get("template", None):
            cmd += " --template {}".format(chat_config.get("template"))
        if chat_config.get("gen_config", None):
            cmd += " --gen_config {}".format(chat_config.get("gen_config"))
        env = os.environ
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, chat_config.get("device")))
        processes.append(Popen(cmd, env=env, shell=True, stdout=PIPE, stderr=STDOUT))
        chat_models[chat_config.get("name")] = AsyncOpenAI(
            api_key="0", base_url="http://127.0.0.1:{}/v1".format(chat_config.get("port"))
        )

    for embed_config in config["embed"]:
        cmd = "python -m imitater.service.embed"
        cmd += " --name {}".format(embed_config.get("name"))
        cmd += " --path {}".format(embed_config.get("path"))
        cmd += " --device {}".format(" ".join(map(str, embed_config.get("device"))))
        cmd += " --port {}".format(embed_config.get("port"))
        if chat_config.get("batch_size", None):
            cmd += " --batch_size {}".format(chat_config.get("batch_size"))
        env = os.environ
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, embed_config.get("device")))
        processes.append(Popen(cmd, env=env, shell=True, stdout=PIPE, stderr=STDOUT))
        embed_models[embed_config.get("name")] = AsyncOpenAI(
            api_key="0", base_url="http://127.0.0.1:{}/v1".format(embed_config.get("port"))
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
    async def create_chat_completion_v1(request: "ChatCompletionRequest"):
        if request.model not in chat_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
        return await create_chat_completion(request, chat_models[request.model])

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
    async def create_embeddings_v1(request: "EmbeddingsRequest"):
        if request.model not in embed_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
        return await create_embeddings(request, embed_models[request.model])

    for process in processes:
        thread = Thread(target=read_message, args=[process])
        thread.start()

    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    launch_server(getattr(args, "config"))
