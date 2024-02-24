from subprocess import Popen
from typing import Optional

from ..utils.generic import jsonify
from .protocol import ChatCompletionMessage, ChatCompletionStreamResponse, ChatCompletionStreamResponseChoice, Finish


def print_subprocess_stdout(process: "Popen") -> None:
    while process.stdout.readable():
        line: bytes = process.stdout.readline()

        if not line:
            break

        print(line.decode("utf-8").strip())


def create_stream_chunk(
    request_id: str,
    model: str,
    delta: "ChatCompletionMessage",
    index: Optional[int] = 0,
    finish_reason: Optional[Finish] = None,
) -> str:
    choice = ChatCompletionStreamResponseChoice(index=index, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=request_id, model=model, choices=[choice])
    return jsonify(chunk)
