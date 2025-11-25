import { X, Sliders } from 'lucide-react'

export default function Settings({ isOpen, onClose, settings, onUpdate }) {
  if (!isOpen) return null

  const handleChange = (key, value) => {
    onUpdate({ ...settings, [key]: value })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="glass-panel w-full max-w-2xl max-h-[90vh] overflow-y-auto m-4">
        {/* Header */}
        <div className="sticky top-0 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sliders size={24} className="text-havoc-500" />
            <h2 className="text-xl font-semibold">Generation Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Temperature */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Temperature
              </label>
              <span className="text-sm text-gray-500">{settings.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={settings.temperature}
              onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-havoc-600"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Focused</span>
              <span>Balanced</span>
              <span>Creative</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Higher values make output more random and creative.
            </p>
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Max Tokens
              </label>
              <span className="text-sm text-gray-500">{settings.maxNewTokens}</span>
            </div>
            <input
              type="range"
              min="50"
              max="2048"
              step="50"
              value={settings.maxNewTokens}
              onChange={(e) => handleChange('maxNewTokens', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-havoc-600"
            />
            <p className="text-xs text-gray-500 mt-2">
              Maximum number of tokens to generate in response.
            </p>
          </div>

          {/* Top P */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Top P (Nucleus Sampling)
              </label>
              <span className="text-sm text-gray-500">{settings.topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={settings.topP}
              onChange={(e) => handleChange('topP', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-havoc-600"
            />
            <p className="text-xs text-gray-500 mt-2">
              Cumulative probability threshold for token selection.
            </p>
          </div>

          {/* Top K */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Top K
              </label>
              <span className="text-sm text-gray-500">{settings.topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={settings.topK}
              onChange={(e) => handleChange('topK', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-havoc-600"
            />
            <p className="text-xs text-gray-500 mt-2">
              Limits sampling to top K most likely tokens.
            </p>
          </div>

          {/* Repetition Penalty */}
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Repetition Penalty
              </label>
              <span className="text-sm text-gray-500">{settings.repetitionPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="1"
              max="2"
              step="0.05"
              value={settings.repetitionPenalty}
              onChange={(e) => handleChange('repetitionPenalty', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-havoc-600"
            />
            <p className="text-xs text-gray-500 mt-2">
              Penalty for repeating tokens. Higher values reduce repetition.
            </p>
          </div>

          {/* Presets */}
          <div className="pt-4 border-t border-gray-800">
            <label className="text-sm font-medium text-gray-300 mb-3 block">
              Presets
            </label>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => onUpdate({
                  temperature: 0.3,
                  topP: 0.85,
                  topK: 30,
                  maxNewTokens: 512,
                  repetitionPenalty: 1.05,
                })}
                className="button-secondary text-sm py-3"
              >
                Precise
              </button>
              <button
                onClick={() => onUpdate({
                  temperature: 0.7,
                  topP: 0.9,
                  topK: 50,
                  maxNewTokens: 512,
                  repetitionPenalty: 1.1,
                })}
                className="button-secondary text-sm py-3"
              >
                Balanced
              </button>
              <button
                onClick={() => onUpdate({
                  temperature: 0.9,
                  topP: 0.95,
                  topK: 100,
                  maxNewTokens: 512,
                  repetitionPenalty: 1.2,
                })}
                className="button-secondary text-sm py-3"
              >
                Creative
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
