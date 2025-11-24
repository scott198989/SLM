import { useState, useRef, useEffect } from 'react'
import { Trash2, Settings as SettingsIcon, Zap, AlertCircle } from 'lucide-react'
import Scene3D from './components/Scene3D'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import Settings from './components/Settings'
import { api } from './lib/api'

const DEFAULT_SETTINGS = {
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  maxNewTokens: 512,
  repetitionPenalty: 1.1,
}

const EXAMPLE_PROMPTS = [
  "What is the derivative of x^2?",
  "Explain the central limit theorem",
  "Design a Box-Behnken DOE for 3 factors",
  "How do I calculate a confidence interval?",
]

function App() {
  const [messages, setMessages] = useState([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [showSettings, setShowSettings] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const abortControllerRef = useRef(null)

  // Check API health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const checkHealth = async () => {
    try {
      await api.health()
      setApiStatus('connected')
      setError(null)
    } catch (err) {
      setApiStatus('error')
      setError('Cannot connect to HAVOC-7B API. Make sure the server is running.')
    }
  }

  const handleSend = async (input) => {
    if (!input.trim() || isGenerating) return

    // Add user message
    const userMessage = { role: 'user', content: input }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)

    // Add empty assistant message for streaming
    const assistantMessage = { role: 'assistant', content: '' }
    setMessages([...newMessages, assistantMessage])
    setIsGenerating(true)
    setError(null)

    try {
      // Use chat endpoint with streaming
      const stream = await api.chat(
        newMessages.map(m => ({ role: m.role, content: m.content })),
        {
          ...settings,
          stream: true,
        }
      )

      let fullResponse = ''

      for await (const token of stream) {
        if (abortControllerRef.current?.aborted) {
          break
        }
        fullResponse += token
        setMessages([...newMessages, { role: 'assistant', content: fullResponse }])
      }
    } catch (err) {
      console.error('Generation error:', err)
      setError(err.message || 'Failed to generate response')
      // Remove the empty assistant message on error
      setMessages(newMessages)
    } finally {
      setIsGenerating(false)
      abortControllerRef.current = null
    }
  }

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setIsGenerating(false)
  }

  const handleClear = () => {
    if (confirm('Clear all messages?')) {
      setMessages([])
    }
  }

  const handleExampleClick = (prompt) => {
    if (!isGenerating) {
      handleSend(prompt)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* 3D Background */}
      <Scene3D />

      {/* Header */}
      <header className="glass-panel border-b border-gray-800/50 px-6 py-4 flex items-center justify-between relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-havoc-600 to-havoc-800 rounded-xl flex items-center justify-center">
            <Zap size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-havoc-400 to-blue-400 bg-clip-text text-transparent">
              HAVOC-7B
            </h1>
            <p className="text-xs text-gray-500">Math, Stats & Engineering AI</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* API Status */}
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === 'connected' ? 'bg-green-500' :
              apiStatus === 'checking' ? 'bg-yellow-500' :
              'bg-red-500'
            }`} />
            <span className="text-gray-400 hidden sm:inline">
              {apiStatus === 'connected' ? 'Connected' :
               apiStatus === 'checking' ? 'Connecting...' :
               'Offline'}
            </span>
          </div>

          {/* Settings Button */}
          <button
            onClick={() => setShowSettings(true)}
            className="button-secondary px-3 py-2"
            title="Settings"
          >
            <SettingsIcon size={18} />
          </button>

          {/* Clear Button */}
          <button
            onClick={handleClear}
            disabled={messages.length === 0}
            className="button-secondary px-3 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto">
          {/* Error Banner */}
          {error && (
            <div className="mb-6 glass-panel border-red-500/50 bg-red-500/10 p-4 flex items-start gap-3">
              <AlertCircle size={20} className="text-red-500 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="font-medium text-red-400">Error</div>
                <div className="text-sm text-gray-300 mt-1">{error}</div>
              </div>
            </div>
          )}

          {/* Welcome Screen */}
          {messages.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-havoc-600 to-havoc-800 rounded-2xl flex items-center justify-center mb-6 animate-float">
                <Zap size={40} className="text-white" />
              </div>
              <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-havoc-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                Welcome to HAVOC-7B
              </h2>
              <p className="text-gray-400 mb-8 max-w-md">
                A specialized language model for mathematics, statistics, and engineering.
                Ask me anything from calculus to Six Sigma.
              </p>

              {/* Example Prompts */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
                {EXAMPLE_PROMPTS.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => handleExampleClick(prompt)}
                    className="glass-panel p-4 text-left hover:bg-gray-800/50 transition-all text-sm"
                  >
                    <div className="text-havoc-400 mb-1 font-medium">Example {index + 1}</div>
                    <div className="text-gray-300">{prompt}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.length > 0 && (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <ChatMessage
                  key={index}
                  message={message}
                  isStreaming={index === messages.length - 1 && isGenerating}
                />
              ))}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <div className="sticky bottom-0 px-4 py-4 bg-gradient-to-t from-gray-950 via-gray-950/95 to-transparent">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSend={handleSend}
            isGenerating={isGenerating}
            onStop={handleStop}
          />
        </div>
      </div>

      {/* Settings Modal */}
      <Settings
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onUpdate={setSettings}
      />
    </div>
  )
}

export default App
