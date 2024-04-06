# Imitater

[![GitHub Code License](https://img.shields.io/github/license/the-seeds/imitater)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/imitater)](https://pypi.org/project/imitater/)

A unified language model server built upon [vllm](https://github.com/vllm-project/vllm) and [infinity](https://github.com/michaelfeil/infinity).

## Usage

### Install

```bash
pip install packaging
pip install -e .
```

### Launch Server

```bash
python -m imitater.service.app -c config/example.yaml
```

<details><summary>Show configuration instruction.</summary>

#### Add an OpenAI model

```yaml
- name: OpenAI model name
- token: OpenAI token
```

#### Add a chat model

```yaml
- name: Display name
- path: Model name on hub or local model path
- device: Device IDs
- port: Port ID
- maxlen: Maximum model length (optional)
- agent_type: Agent type (optional) {react, aligned}
- template: Template jinja file (optional)
- gen_config: Generation config folder (optional)
```

#### Add an embedding model

```yaml
- name: Display name
- path: Model name on hub or local model path
- device: Device IDs (does not support multi-gpus)
- port: Port ID
- batch_size: Batch size (optional)
```

</details>

> [!NOTE]
> [Chat template](https://huggingface.co/docs/transformers/chat_templating) is required for the chat models.
>
> Use `export USE_MODELSCOPE_HUB=1` to download model from modelscope.

### Test Server

```bash
python tests/test_openai.py -c config/example.yaml
```

### Roadmap

- [ ] Response choices.
- [ ] Rerank model support.
