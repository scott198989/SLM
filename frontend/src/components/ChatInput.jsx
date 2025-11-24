import { useState } from 'react'
import { Send, Square } from 'lucide-react'

export default function ChatInput({ onSend, isGenerating, onStop }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !isGenerating) {
      onSend(input)
      setInput('')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="glass-panel p-4 shadow-2xl">
        <div className="flex gap-3 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask HAVOC-7B anything about math, statistics, or engineering..."
            className="flex-1 bg-gray-900/80 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-havoc-600 focus:border-transparent transition-all resize-none"
            rows="3"
            disabled={isGenerating}
          />

          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              className="bg-red-600 hover:bg-red-700 text-white p-3 rounded-xl transition-all duration-200 ease-in-out active:scale-95 flex-shrink-0"
              title="Stop generation"
            >
              <Square size={20} fill="currentColor" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="button-primary p-3 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
              title="Send message"
            >
              <Send size={20} />
            </button>
          )}
        </div>

        <div className="mt-2 text-xs text-gray-500">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </form>
  )
}
