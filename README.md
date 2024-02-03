# Imitater

[![GitHub Code License](https://img.shields.io/github/license/the-seeds/imitater)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/imitater)](https://pypi.org/project/imitater/)

A unified language model server built upon [vllm](https://github.com/vllm-project/vllm) and [infinity](https://github.com/michaelfeil/infinity).

## Usage

### Install


```bash
pip install -U imitater
```

### Launch Server

```bash
python -m imitater.service.app -c config/example.yaml
```

> [!NOTE]
> [Chat template](https://huggingface.co/docs/transformers/chat_templating) is required for the chat models.

### Test Server

```bash
python tests/test_openai.py -c config/example.yaml
```
