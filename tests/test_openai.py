import argparse
from enum import Enum, unique
from typing import Any, Dict

import click
import yaml
from openai import OpenAI


try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")


@unique
class Action(str, Enum):
    CHAT = "chat"
    EMBED = "embed"
    TOOL = "tool"


def test_chat(client: "OpenAI", query: str) -> None:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="gpt-3.5-turbo",
        stream=True,
    )
    print("Assistant: ", end="")
    for chunk in filter(lambda p: p.choices[0].delta.content is not None, stream):
        print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def test_embed(client: "OpenAI", query: str) -> None:
    data = client.embeddings.create(input=query, model="text-embedding-ada-002", encoding_format="float").data
    for embedding in data:
        print(embedding.embedding)


def test_tool(client: "OpenAI") -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        model="gpt-3.5-turbo",
        tools=tools,
    )
    print(result.choices[0].message)
    result = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "What is the weather like in Boston?"},
            {
                "role": "function",
                "content": """{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}""",
            },
            {"role": "tool", "content": """{"temperature": 22, "unit": "celsius", "description": "Sunny"}"""},
        ],
        model="gpt-3.5-turbo",
        tools=tools,
    )
    print(result.choices[0].message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    with open(getattr(args, "config"), "r", encoding="utf-8") as f:
        config: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    client = OpenAI(
        api_key="0",
        base_url="http://{host}:{port}/v1".format(
            host=config["service"].get("host", "localhost"),
            port=config["service"].get("port", 8000),
        ),
    )
    action = click.prompt("Action", type=click.Choice([act.value for act in Action]))
    if action == Action.TOOL:
        test_tool(client)
    else:
        while True:
            query = click.prompt("User", type=str)
            if query == "exit":
                break

            if action == Action.CHAT:
                test_chat(client, query)
            elif action == Action.EMBED:
                test_embed(client, query)
