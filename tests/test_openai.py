import enum

import click
from dotenv import load_dotenv
from openai import OpenAI


try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")


@enum.unique
class Action(str, enum.Enum):
    CHAT = "chat"
    EMBED = "embed"


def test_chat(query: str):
    client = OpenAI()
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": query}], model="gpt-3.5-turbo", stream=True
    )
    print("Assistant: ", end="")
    for chunk in filter(lambda p: p.choices[0].delta.content is not None, stream):
        print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def test_embed(query: str):
    client = OpenAI()
    data = client.embeddings.create(input=query, model="text-embedding-ada-002").data
    for embedding in data:
        print(embedding.embedding)


if __name__ == "__main__":
    load_dotenv()
    action = click.prompt("Action", type=click.Choice([act.value for act in Action]))
    test_func = test_chat if action == Action.CHAT else test_embed

    while True:
        query = click.prompt("User", type=str)
        if query == "exit":
            break

        test_func(query)
