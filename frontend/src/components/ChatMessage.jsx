import { User, Bot } from 'lucide-react'
import MarkdownMessage from './MarkdownMessage'

export default function ChatMessage({ message, isStreaming }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'} mb-6`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
        isUser ? 'bg-havoc-600' : 'bg-gray-800'
      }`}>
        {isUser ? (
          <User size={20} className="text-white" />
        ) : (
          <Bot size={20} className="text-havoc-400" />
        )}
      </div>

      {/* Message content */}
      <div className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}>
        <div className="flex items-center gap-2 mb-2">
          <span className="font-semibold text-sm">
            {isUser ? 'You' : 'HAVOC-7B'}
          </span>
          {!isUser && isStreaming && (
            <div className="flex gap-1">
              <div className="w-1.5 h-1.5 bg-havoc-500 rounded-full thinking-dot"></div>
              <div className="w-1.5 h-1.5 bg-havoc-500 rounded-full thinking-dot"></div>
              <div className="w-1.5 h-1.5 bg-havoc-500 rounded-full thinking-dot"></div>
            </div>
          )}
        </div>

        {message.content ? (
          <MarkdownMessage content={message.content} />
        ) : (
          <div className="text-gray-500 italic">Thinking...</div>
        )}
      </div>
    </div>
  )
}
