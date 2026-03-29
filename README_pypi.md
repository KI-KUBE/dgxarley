[![mypy and pytests](https://github.com/vroomfondel/dgxarley/actions/workflows/mypynpytests.yml/badge.svg)](https://github.com/vroomfondel/dgxarley/actions/workflows/mypynpytests.yml)
[![black-lint](https://github.com/vroomfondel/dgxarley/actions/workflows/checkblack.yml/badge.svg)](https://github.com/vroomfondel/dgxarley/actions/workflows/checkblack.yml)
[![Cumulative Clones](https://img.shields.io/endpoint?logo=github&url=https://gist.githubusercontent.com/vroomfondel/3e9b5513788bd4a8af48560724376534/raw/dgxarley_clone_count.json)](https://github.com/vroomfondel/dgxarley)


# dgxarley

Tooling for the DGX Arley K3s inference cluster — integration tests, streaming utilities, and CLI entry points for SGLang, Ollama, and OpenWebUI services.

## What's included

### CLI tools

| Command | Description |
|---------|-------------|
| `sglang-raw` | Interactive SSE stream viewer with dual-panel Rich display (interpreted output + raw JSON chunks) |
| `sglang-test` | Direct SGLang client with sequential and parallel load testing (live Rich TUI) |
| `openwebui-test` | OpenWebUI / LLM client with preset management and streaming |
| `ollama-test` | Ollama API health, model, embedding, and chat completions tests |

### Libraries

| Module | Description |
|--------|-------------|
| `dgxarley.integration.repetition_detector` | Offline n-gram, sentence, and loop repetition analysis for completed LLM outputs |
| `dgxarley.integration.streaming_repetition_guard` | Real-time repetition detection for token streams with configurable thresholds |

## Installation

```bash
pip install dgxarley
```

## Quick start

```python
from dgxarley.integration.repetition_detector import detect_repetition

report = detect_repetition(llm_output)
print(report.summary())
# [LOW] score=0.12 — N-Gram 'this is a test' x2
```

```python
from dgxarley.integration.streaming_repetition_guard import RepetitionGuard

guard = RepetitionGuard()
for chunk in llm_stream:
    token = chunk.choices[0].delta.content or ""
    result = guard.feed(token)
    if result.should_stop:
        print(f"STOP: {result.reason}")
        break
```

## Requirements

- Python >= 3.14

## Source & documentation

Full documentation, network architecture, and Ansible playbooks: [GitHub](https://github.com/vroomfondel/dgxarley)

## License
This project is licensed under the LGPL where applicable/possible — see [LICENSE.md](https://github.com/vroomfondel/dgxarley/blob/main/LICENSE.md). Some files/parts may use other licenses: [MIT](https://github.com/vroomfondel/dgxarley/blob/main/LICENSEMIT.md) | [GPL](https://github.com/vroomfondel/dgxarley/blob/main/LICENSEGPL.md) | [LGPL](https://github.com/vroomfondel/dgxarley/blob/main/LICENSELGPL.md). Always check per‑file headers/comments.


## Authors
- Repo owner (primary author)
- Additional attributions are noted inline in code comments


## Acknowledgments
- Inspirations and snippets are referenced in code comments where appropriate.


## ⚠️ Note

This is a development/experimental project. For production use, review security settings, customize configurations, and test thoroughly in your environment. Provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. Use at your own risk.


