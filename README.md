# Thesis AI

An artificial intelligence system that summarizes academic papers and research articles

## Installation

```sh
poetry install --no-root
```

## Setup

### 1. Copy .env.sample to .env

```sh
cp .env.sample .env
```

<br />

### 2. Fill in your information in the .env file with appropriate values

```yaml
BASE_URL=http://localhost:11434
EMBEDDING_MODEL_NAME=nomic-embed-text
MODEL_NAME=llama3.3
```

<br />

### 3. Add your academic paper PDFs to the assets folder

<br />

---

## Run

```sh
poetry run python -m app
```

## Quit

Say `/bye` or `quit` or `exit`

---

## LICENSE

[MIT](https://github.com/jwkwon0817/thesis-ai/blob/main/LICENSE)
