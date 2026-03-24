"use client";

import { useState, KeyboardEvent, useRef, useEffect } from "react";
import { ArrowUp, Link as LinkIcon, X } from "lucide-react";
import { uploadDocument } from "@/lib/api";
import clsx from "clsx";

interface InputAreaProps {
  onSend: (message: string) => void;
  selectedDocCount: number;
  onClearDocs: () => void;
  isLoading: boolean;
  onUploadSuccess: () => void;
}

export default function InputArea({ onSend, selectedDocCount, onClearDocs, isLoading, onUploadSuccess }: InputAreaProps) {
  const [content, setContent] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [content]);

  const handleSend = () => {
    if (!content.trim() || isLoading) return;
    onSend(content.trim());
    setContent("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    try {
      await uploadDocument(file);
      onUploadSuccess();
    } catch (err) {
      alert("上传失败: " + err);
    } finally {
      e.target.value = "";
    }
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto px-4 pb-6 mt-2">
      
      {/* Dynamic Documentation Hint Pill */}
      {selectedDocCount > 0 && (
        <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 flex items-center justify-center">
          <div className="flex items-center gap-2 bg-blue-50 text-blue-600 px-4 py-1.5 rounded-full shadow-sm border border-blue-100 text-xs font-semibold animate-in fade-in slide-in-from-bottom-2">
            已选定 {selectedDocCount} 篇文档作为上下文
            <button 
              onClick={onClearDocs}
              className="ml-2 hover:bg-blue-100 rounded-full p-0.5 transition-colors"
              title="清空选择"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Main Input Box */}
      <div className="relative flex items-end w-full bg-white border border-gray-200 rounded-2xl shadow-sm focus-within:ring-1 focus-within:ring-blue-500 focus-within:border-blue-500 overflow-hidden transition-all pl-3">
        
        {/* Upload Button */}
        <label className="mb-2.5 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-xl cursor-pointer transition-colors" title="上传知识文档">
          <LinkIcon size={20} />
          <input type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} />
        </label>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="给 RAG 助手发送消息..."
          className="flex-1 max-h-[200px] min-h-[44px] py-3 px-3 bg-transparent border-none focus:outline-none focus:ring-0 resize-none custom-scrollbar text-[15px]"
          rows={1}
        />

        {/* Send Button */}
        <div className="mb-2 mr-2 ml-2">
          <button
            onClick={handleSend}
            disabled={!content.trim() || isLoading}
            className={clsx(
              "w-9 h-9 flex items-center justify-center rounded-xl transition-all shadow-sm",
              content.trim() && !isLoading
                ? "bg-blue-600 text-white hover:bg-blue-700" 
                : "bg-gray-100 text-gray-400 cursor-not-allowed"
            )}
          >
            <ArrowUp size={18} strokeWidth={2.5} />
          </button>
        </div>
      </div>
      
      <div className="text-center mt-2 text-[11px] text-gray-400">
        RAG 助手可能会生成错误信息，请核实重要信息。支持多选左侧文档来限定回答范围。
      </div>
    </div>
  );
}
