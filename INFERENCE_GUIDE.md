# HAVOC-7B Inference API Guide

## Overview

The inference API is a production-ready FastAPI server for serving the trained HAVOC-7B model. It provides REST endpoints for text completion and chat, with support for streaming responses.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[inference]"
```

This installs:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)

### 2. Start the Server

```bash
# Basic usage (with random weights for testing)
python scripts/serve.py

# With trained checkpoint
python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10000

# Custom port
python scripts/serve.py --port 8080

# With config file
python scripts/serve.py --config configs/inference/default_inference.yaml
```

The server will start on `http://localhost:8000` by default.

### 3. Test the API

**Health check:**
```bash
curl http://localhost:8000/health
```

**Text completion:**
```bash
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the derivative of x^2?",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain the central limit theorem"}
    ],
    "max_new_tokens": 200
  }'
```

**Interactive API docs:**
Visit `http://localhost:8000/docs` for Swagger UI with interactive API testing.

---

## API Endpoints

### `GET /`

Root endpoint with basic information.

**Response:**
```json
{
  "message": "HAVOC-7B Inference API",
  "version": "0.1.0",
  "endpoints": {
    "completion": "/completion",
    "chat": "/chat",
    "health": "/health"
  }
}
```

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "havoc-7b",
  "device": "cuda",
  "timestamp": 1234567890.123
}
```

---

### `POST /completion`

Generate text from a prompt.

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "stream": false,
  "stop_sequences": ["\n\n", "END"]
}
```

**Parameters:**
- `prompt` (required): Input text prompt
- `max_new_tokens` (default: 512): Maximum tokens to generate
- `temperature` (default: 0.7): Sampling temperature (0.0-2.0, higher = more random)
- `top_p` (default: 0.9): Nucleus sampling threshold (0.0-1.0)
- `top_k` (default: 50): Top-k sampling threshold
- `repetition_penalty` (default: 1.1): Penalty for repeating tokens (1.0 = no penalty)
- `do_sample` (default: true): Use sampling vs greedy decoding
- `stream` (default: false): Stream response token-by-token
- `stop_sequences` (optional): List of strings that stop generation

**Response (non-streaming):**
```json
{
  "text": "Paris, the city of lights and romance...",
  "prompt": "The capital of France is",
  "model": "havoc-7b",
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 50,
    "total_tokens": 56
  }
}
```

**Response (streaming):**
Server-sent events (SSE) format:
```
data: Paris
data: ,
data:  the
data:  city
...
data: [DONE]
```

---

### `POST /chat`

Generate chat responses from a conversation.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is the quadratic formula?"},
    {"role": "assistant", "content": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a"},
    {"role": "user", "content": "Can you explain when to use it?"}
  ],
  "max_new_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

**Parameters:**
Same as `/completion`, plus:
- `messages` (required): List of chat messages with `role` and `content`

**Roles:**
- `system`: System instructions/context
- `user`: User messages
- `assistant`: Assistant responses

**Response (non-streaming):**
```json
{
  "message": {
    "role": "assistant",
    "content": "The quadratic formula is used to solve quadratic equations..."
  },
  "model": "havoc-7b",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 150,
    "total_tokens": 195
  }
}
```

**Response (streaming):**
Same SSE format as `/completion`.

---

## Generation Parameters

### Temperature

Controls randomness in sampling:
- `0.0-0.3`: Very focused, deterministic (good for factual/technical tasks)
- `0.5-0.7`: Balanced (default, good for most tasks)
- `0.8-1.0`: Creative, diverse (good for creative writing)
- `> 1.0`: Very random (experimental)

### Top-p (Nucleus Sampling)

Keeps only tokens with cumulative probability above threshold:
- `0.9`: Standard (keeps most likely 90% of probability mass)
- `0.95`: More diverse
- `0.85`: More focused

### Top-k

Keeps only the k most likely tokens:
- `50`: Standard
- `10-30`: More focused
- `100+`: More diverse

### Repetition Penalty

Reduces probability of repeating tokens:
- `1.0`: No penalty
- `1.1`: Slight penalty (default)
- `1.2-1.5`: Stronger penalty (use if model repeats too much)

### Sampling vs Greedy

- `do_sample: true`: Use sampling (random, diverse)
- `do_sample: false`: Greedy decoding (deterministic, always picks most likely token)

---

## Configuration

### Default Configuration

See `configs/inference/default_inference.yaml` for all options.

### Command-Line Overrides

```bash
python scripts/serve.py \
  --checkpoint checkpoints/checkpoint_step_10000 \
  --host 0.0.0.0 \
  --port 8080 \
  --device cuda
```

### Environment-Specific Settings

**Development:**
```bash
python scripts/serve.py --reload
```

**Production:**
```bash
# Use gunicorn or production ASGI server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker havoc_inference.server:app
```

---

## Client Examples

### Python (requests)

```python
import requests

# Completion
response = requests.post(
    "http://localhost:8000/completion",
    json={
        "prompt": "What is Six Sigma?",
        "max_new_tokens": 100,
        "temperature": 0.5,
    }
)
print(response.json()["text"])

# Chat
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "messages": [
            {"role": "user", "content": "Explain ANOVA"}
        ],
        "max_new_tokens": 200,
    }
)
print(response.json()["message"]["content"])
```

### Python (streaming)

```python
import requests

response = requests.post(
    "http://localhost:8000/completion",
    json={
        "prompt": "Write a factorial function in Python:",
        "stream": True,
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            token = line[6:]
            if token == '[DONE]':
                break
            print(token, end='', flush=True)
```

### JavaScript (fetch)

```javascript
// Completion
const response = await fetch('http://localhost:8000/completion', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'What is Design of Experiments?',
    max_new_tokens: 150,
  })
});
const data = await response.json();
console.log(data.text);

// Chat
const chatResponse = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    messages: [
      {role: 'user', content: 'Explain linear regression'}
    ]
  })
});
const chatData = await chatResponse.json();
console.log(chatData.message.content);
```

### JavaScript (streaming)

```javascript
const response = await fetch('http://localhost:8000/completion', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'Write a poem about mathematics:',
    stream: true,
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const token = line.slice(6);
      if (token === '[DONE]') break;
      process.stdout.write(token);
    }
  }
}
```

### cURL

```bash
# Completion
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate the mean of 1, 2, 3, 4, 5:",
    "temperature": 0.3,
    "max_new_tokens": 50
  }'

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a statistics expert."},
      {"role": "user", "content": "What is p-value?"}
    ],
    "temperature": 0.5
  }'

# Streaming
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "prompt": "Explain Bayes theorem:",
    "stream": true
  }'
```

---

## Performance Optimization

### Batch Processing

The server supports batch processing (up to `max_batch_size` requests concurrently).

### GPU Memory

Monitor GPU memory:
```bash
nvidia-smi -l 1
```

If running out of memory:
- Reduce `max_batch_size` in config
- Enable mixed precision (`use_amp: true`)
- Use bfloat16 instead of float16

### Caching

The model uses KV-cache for efficient generation. Past key-values are cached during streaming.

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -e ".[inference]"

EXPOSE 8000

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t havoc-7b-api .
docker run -p 8000:8000 --gpus all havoc-7b-api
```

### Production ASGI Server

For production, use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn

gunicorn \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  havoc_inference.server:app
```

### Load Balancing

For high traffic, use a load balancer (nginx, HAProxy) to distribute requests across multiple instances.

---

## Troubleshooting

**Server won't start:**
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port: `--port 8080`

**Out of memory:**
- Enable mixed precision: `use_amp: true` in config
- Reduce batch size: `max_batch_size: 4`
- Use smaller model or reduce `max_seq_len`

**Slow generation:**
- Enable mixed precision
- Use GPU instead of CPU
- Check if model is on correct device: `device: cuda`

**Model outputs gibberish:**
- Likely using random weights (no checkpoint loaded)
- Load trained checkpoint: `--checkpoint checkpoints/checkpoint_step_N`
- Adjust generation parameters (temperature, top_p)

---

## Next Steps

1. **Train the model**: Use `scripts/train.py` to train HAVOC-7B
2. **Load checkpoint**: Point server to trained checkpoint
3. **Fine-tune**: Fine-tune on domain-specific data
4. **Integrate RAG**: Add retrieval for grounded responses
5. **Build UI**: Create chat UI (see next section of project)

---

## API Reference

Full API documentation available at: `http://localhost:8000/docs` (Swagger UI)

Alternative documentation: `http://localhost:8000/redoc` (ReDoc)

---

**Ready to serve!** 🚀

```bash
python scripts/serve.py --checkpoint checkpoints/your_checkpoint
```
