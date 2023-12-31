# Imitater

[![GitHub Code License](https://img.shields.io/github/license/the-seeds/imitater)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/imitater)](https://pypi.org/project/imitater/)

## Usage

Create a `.env` file in the root directory:

```
.
├── src
└── .env
```

```
# imitater
CHAT_MODEL=hiyouga/Qwen-14B-Chat-LLaMAfied
EMBED_MODEL=BAAI/bge-small-zh-v1.5
EMBED_BATCH_SIZE=16
ENABLE_ATTN_BIAS=1

# tests
OPENAI_BASE_URL=http://192.168.0.1:8000/v1
OPENAI_API_KEY=0
```

> [!NOTE]
> [Chat template](https://huggingface.co/docs/transformers/chat_templating) is required for the chat models.

## Launch Server

```bash
python src/launch.py
```

## Test Server

```bash
python tests/test_openai.py
```