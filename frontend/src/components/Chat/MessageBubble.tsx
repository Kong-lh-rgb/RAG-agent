"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css"; // Required for LaTeX styling
import { UserRound, Bot } from "lucide-react";
import clsx from "clsx";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
}

export default function MessageBubble({ role, content }: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={clsx("flex gap-4 w-full py-6", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
          <Bot size={18} className="text-blue-600" />
        </div>
      )}
      
      <div 
        className={clsx(
          "max-w-[85%] rounded-2xl px-5 py-3 text-[15px] leading-relaxed",
          isUser 
            ? "bg-blue-600 text-white rounded-br-sm" 
            : "bg-white border border-gray-100 shadow-sm text-gray-800 rounded-bl-sm"
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : (
          <div className="markdown-prose prose-sm md:prose-base max-w-none text-gray-800">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3" {...props} />,
                ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3" {...props} />,
                li: ({node, ...props}) => <li className="mb-1" {...props} />,
                table: ({node, ...props}) => (
                  <div className="overflow-x-auto mb-4">
                    <table className="min-w-full divide-y divide-gray-200 border border-gray-200 rounded-md" {...props} />
                  </div>
                ),
                th: ({node, ...props}) => <th className="bg-gray-50 px-4 py-2 text-left font-semibold text-gray-700 text-sm border-b" {...props} />,
                td: ({node, ...props}) => <td className="px-4 py-2 text-sm border-b border-gray-100" {...props} />,
                pre: ({node, ...props}) => (
                  <pre className="bg-gray-800 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4 text-sm font-mono" {...props} />
                ),
                code: ({node, inline, ...props}: any) => 
                  inline 
                    ? <code className="bg-gray-100 text-pink-600 px-1 py-0.5 rounded text-sm font-mono" {...props} />
                    : <code {...props} />
              }}
            >
              {content || "..."}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-1">
          <UserRound size={18} className="text-gray-600" />
        </div>
      )}
    </div>
  );
}
