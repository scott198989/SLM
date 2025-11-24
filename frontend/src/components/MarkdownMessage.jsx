import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import 'katex/dist/katex.min.css'

export default function MarkdownMessage({ content }) {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // Custom code block rendering
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '')
            return !inline ? (
              <code className={className} {...props}>
                {children}
              </code>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            )
          },
          // Custom link rendering
          a({ node, children, href, ...props }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-havoc-400 hover:text-havoc-300 underline"
                {...props}
              >
                {children}
              </a>
            )
          },
          // Custom table rendering
          table({ node, children, ...props }) {
            return (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-700" {...props}>
                  {children}
                </table>
              </div>
            )
          },
          th({ node, children, ...props }) {
            return (
              <th className="px-4 py-2 bg-gray-800 text-left text-sm font-semibold" {...props}>
                {children}
              </th>
            )
          },
          td({ node, children, ...props }) {
            return (
              <td className="px-4 py-2 border-t border-gray-800 text-sm" {...props}>
                {children}
              </td>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
