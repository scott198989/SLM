const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export class HavocAPI {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`)
    if (!response.ok) {
      throw new Error('API health check failed')
    }
    return await response.json()
  }

  async completion(prompt, options = {}) {
    const {
      maxNewTokens = 512,
      temperature = 0.7,
      topP = 0.9,
      topK = 50,
      repetitionPenalty = 1.1,
      doSample = true,
      stream = false,
      stopSequences = null,
    } = options

    const body = {
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repetitionPenalty,
      do_sample: doSample,
      stream,
    }

    if (stopSequences) {
      body.stop_sequences = stopSequences
    }

    if (stream) {
      return this._streamCompletion(body)
    }

    const response = await fetch(`${this.baseUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Completion request failed')
    }

    return await response.json()
  }

  async chat(messages, options = {}) {
    const {
      maxNewTokens = 512,
      temperature = 0.7,
      topP = 0.9,
      topK = 50,
      repetitionPenalty = 1.1,
      doSample = true,
      stream = false,
      stopSequences = null,
    } = options

    const body = {
      messages,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repetitionPenalty,
      do_sample: doSample,
      stream,
    }

    if (stopSequences) {
      body.stop_sequences = stopSequences
    }

    if (stream) {
      return this._streamChat(body)
    }

    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Chat request failed')
    }

    return await response.json()
  }

  async *_streamCompletion(body) {
    const response = await fetch(`${this.baseUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Completion request failed')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              return
            }
            yield data
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  async *_streamChat(body) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Chat request failed')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              return
            }
            yield data
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }
}

export const api = new HavocAPI()
