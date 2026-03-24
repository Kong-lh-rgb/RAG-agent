"use client";

import { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import Sidebar from "@/components/Sidebar";
import MessageBubble from "@/components/Chat/MessageBubble";
import InputArea from "@/components/Chat/InputArea";
import { streamChat, ChatMessage } from "@/lib/api";

export default function Home() {
  const [sessionId, setSessionId] = useState<string>(() => uuidv4());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [refreshSidebarCount, setRefreshSidebarCount] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    setSessionId(uuidv4());
    setMessages([]);
    setSelectedDocIds([]);
  };

  const handleSend = async (content: string) => {
    if (!content.trim() || isStreaming) return;

    // Add user message
    const userMsg: ChatMessage = { role: "user", content };
    setMessages((prev) => [...prev, userMsg, { role: "assistant", content: "" }]);
    setIsStreaming(true);

    let assistantContent = "";

    try {
      await streamChat(
        content,
        sessionId,
        selectedDocIds,
        (token) => {
          assistantContent += token;
          setMessages((prev) => {
            const newMsgs = [...prev];
            // Guard against race conditions (e.g. user starts a new chat mid-stream).
            if (newMsgs.length === 0) {
              return [{ role: "assistant", content: assistantContent }];
            }

            const lastIndex = newMsgs.length - 1;
            if (newMsgs[lastIndex].role !== "assistant") {
              newMsgs.push({ role: "assistant", content: assistantContent });
            } else {
              newMsgs[lastIndex] = { role: "assistant", content: assistantContent };
            }
            return newMsgs;
          });
        },
        undefined,
        () => {
          setIsStreaming(false);
        },
        (err) => {
          console.error("Stream error:", err);
          setIsStreaming(false);
        }
      );
    } catch (e) {
      console.error(e);
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-white overflow-hidden font-sans">
      <Sidebar 
        selectedDocIds={selectedDocIds} 
        onSelectionChange={setSelectedDocIds} 
        onNewChat={handleNewChat}
        refreshTrigger={refreshSidebarCount}
      />
      
      <main className="flex-1 flex flex-col h-full bg-gray-50/30">
        {/* Header (optional minimal branding) */}
        <header className="h-14 border-b border-gray-100 flex items-center px-6 shrink-0 bg-white/80 backdrop-blur-md">
          <h1 className="font-semibold text-gray-800 text-sm tracking-wide">RAG ASSISTANT</h1>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-sm">
                 <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800 tracking-tight mb-2">如何为您提供帮助？</h2>
              <p className="text-gray-500 max-w-md text-sm">
                上传您的企业知识文档，通过智能 RAG 后端快速理解长篇报告、分析数据或解答疑问。
              </p>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto w-full px-4 py-6">
              {messages.map((msg, i) => (
                <MessageBubble key={i} role={msg.role} content={msg.content} />
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="shrink-0 bg-gradient-to-t from-white via-white/80 to-transparent pt-6">
          <InputArea 
            onSend={handleSend}
            selectedDocCount={selectedDocIds.length}
            onClearDocs={() => setSelectedDocIds([])}
            isLoading={isStreaming}
            onUploadSuccess={() => setRefreshSidebarCount(c => c + 1)}
          />
        </div>
      </main>
    </div>
  );
}
