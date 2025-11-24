# HAVOC-7B Chat UI

A modern, polished chat interface for interacting with the HAVOC-7B language model. Built with React, Vite, Three.js, and Tailwind CSS.

## Features

### 🎨 Modern UI Design
- **3D Animated Background**: Subtle floating geometric shapes and particles using Three.js/react-three-fiber
- **Glass morphism**: Beautiful glassmorphic panels with backdrop blur effects
- **Dark theme**: Eye-friendly dark color scheme with HAVOC brand colors
- **Responsive**: Works on desktop, tablet, and mobile

### 💬 Chat Experience
- **Real-time streaming**: Token-by-token streaming of responses
- **Markdown rendering**: Full markdown support including:
  - Code syntax highlighting (with `highlight.js`)
  - LaTeX math rendering (with KaTeX)
  - Tables, lists, links, and more
- **Message history**: Persistent chat history during session
- **Example prompts**: Quick-start examples for common use cases

### ⚙️ Customization
- **Generation settings panel**: Adjust all generation parameters:
  - Temperature (0.0 - 2.0)
  - Max tokens (50 - 2048)
  - Top-p nucleus sampling
  - Top-k sampling
  - Repetition penalty
- **Presets**: Quick presets for Precise, Balanced, and Creative modes

### 🔌 API Integration
- Connects to HAVOC-7B inference server (FastAPI backend)
- Real-time health monitoring
- Streaming support via Server-Sent Events (SSE)
- Error handling and user feedback

## Tech Stack

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **Three.js + react-three-fiber**: 3D graphics
- **@react-three/drei**: Three.js helpers
- **Tailwind CSS**: Utility-first styling
- **react-markdown**: Markdown rendering
- **KaTeX**: Math rendering
- **highlight.js**: Code syntax highlighting
- **Lucide React**: Beautiful icons

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The UI will be available at `http://localhost:3000`

### 3. Configure API Endpoint (Optional)

Create a `.env` file if you need to change the API URL:

```env
VITE_API_URL=http://localhost:8000
```

Default is `http://localhost:8000` (same machine as dev server).

### 4. Start the Backend

In a separate terminal, start the HAVOC-7B inference server:

```bash
cd ..
python scripts/serve.py --checkpoint checkpoints/your_checkpoint
```

## Development

### Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Scene3D.jsx          # 3D background with Three.js
│   │   ├── ChatMessage.jsx      # Individual message component
│   │   ├── ChatInput.jsx        # Message input with send button
│   │   ├── MarkdownMessage.jsx  # Markdown/code/math renderer
│   │   └── Settings.jsx         # Settings modal
│   ├── lib/
│   │   └── api.js               # API client for inference server
│   ├── App.jsx                  # Main application component
│   ├── main.jsx                 # Entry point
│   └── index.css                # Global styles + Tailwind
├── public/                      # Static assets
├── index.html                   # HTML template
├── vite.config.js               # Vite configuration
├── tailwind.config.js           # Tailwind configuration
└── package.json                 # Dependencies
```

### Available Scripts

```bash
# Development server with hot reload
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Building for Production

### 1. Build

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### 2. Preview Build

```bash
npm run preview
```

Test the production build locally before deployment.

### 3. Deploy

The `dist/` directory contains static files ready to deploy to:
- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop `dist/` folder
- **GitHub Pages**: Push `dist/` to `gh-pages` branch
- **Any static host**: Upload `dist/` contents

**Note**: Update `VITE_API_URL` in production to point to your hosted API.

## Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```js
theme: {
  extend: {
    colors: {
      'havoc': {
        // Your custom color palette
      }
    }
  }
}
```

### 3D Background

Modify `src/components/Scene3D.jsx` to:
- Add/remove shapes
- Change colors, sizes, speeds
- Adjust particle count
- Enable/disable effects

### Settings Presets

Add custom presets in `src/components/Settings.jsx`:

```jsx
<button onClick={() => onUpdate({
  temperature: 0.5,
  topP: 0.9,
  // ... your preset
})}>
  My Preset
</button>
```

## API Integration

The UI connects to the HAVOC-7B inference server. Make sure the server is running:

```bash
python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10000
```

The UI uses these endpoints:
- `GET /health` - Health check
- `POST /chat` - Chat completion (with streaming)

## Troubleshooting

**UI won't connect to API:**
- Check that the inference server is running
- Verify the API URL in `.env` or `api.js`
- Check browser console for CORS errors

**3D background not rendering:**
- Ensure WebGL is supported in your browser
- Check browser console for Three.js errors
- Try disabling the 3D background by commenting out `<Scene3D />` in `App.jsx`

**Markdown/code not rendering correctly:**
- Check that `highlight.js` CSS is loaded
- Verify KaTeX CSS is imported
- Check browser console for errors

**Build fails:**
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`
- Check Node.js version (requires Node 18+)

## Browser Support

- **Chrome/Edge**: ✅ Full support
- **Firefox**: ✅ Full support
- **Safari**: ✅ Full support (macOS/iOS)
- **Mobile**: ✅ Responsive design

Requires:
- ES6+ JavaScript
- WebGL (for 3D background)
- Modern CSS (backdrop-filter, etc.)

## Performance

### Optimization Tips

1. **3D Background**: Can be disabled for better performance on low-end devices
2. **Streaming**: Reduces perceived latency vs non-streaming
3. **Code splitting**: Vite automatically code-splits for optimal loading
4. **Asset optimization**: Images and fonts are optimized during build

### Lighthouse Scores (Target)

- Performance: 90+
- Accessibility: 95+
- Best Practices: 95+
- SEO: 90+

## Contributing

To add new features:

1. Create components in `src/components/`
2. Add styles using Tailwind utility classes
3. Update `App.jsx` to integrate the component
4. Test locally with `npm run dev`
5. Build and verify with `npm run build && npm run preview`

## License

Apache 2.0 (same as parent project)

---

**Ready to chat with HAVOC-7B!** 🚀

```bash
npm install && npm run dev
```
