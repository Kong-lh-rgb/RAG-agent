"use client";

import { useEffect, useState } from "react";
import { Sparkles, RefreshCcw, FileText, Upload, UserRound, Trash2 } from "lucide-react";
import { fetchDocuments, DocumentItem, uploadDocument } from "@/lib/api";
import { format } from "date-fns";

interface SidebarProps {
  selectedDocIds: string[];
  onSelectionChange: (ids: string[]) => void;
  onNewChat: () => void;
  refreshTrigger?: number;
}

export default function Sidebar({ selectedDocIds, onSelectionChange, onNewChat, refreshTrigger = 0 }: SidebarProps) {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const loadDocs = async () => {
    setLoading(true);
    try {
      const data = await fetchDocuments();
      setDocs(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocs();
  }, [refreshTrigger]);

  const toggleDoc = (id: string) => {
    if (selectedDocIds.includes(id)) {
      onSelectionChange(selectedDocIds.filter(v => v !== id));
    } else {
      onSelectionChange([...selectedDocIds, id]);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true);
    try {
      await uploadDocument(file);
      await loadDocs(); // reload after upload
    } catch (err) {
      alert("上传失败: " + err);
    } finally {
      setUploading(false);
      e.target.value = ""; // reset input
    }
  };

  return (
    <div className="w-72 h-screen bg-[#F8F9FA] border-r border-gray-200 flex flex-col p-4 text-sm font-medium text-gray-700">
      {/* New Chat Button */}
      <button 
        onClick={onNewChat}
        className="w-full flex items-center gap-2 px-3 py-2.5 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors shadow-sm mb-6 font-semibold"
      >
        <Sparkles size={18} className="text-blue-500" />
        新建对话
      </button>

      {/* Docs Header */}
      <div className="flex items-center justify-between mb-3 px-1 text-xs font-bold text-gray-500 uppercase tracking-wider">
        <span>知识库文档</span>
        <div className="flex gap-2">
          <label className="cursor-pointer hover:text-gray-800 transition-colors" title="上传文档">
            <Upload size={14} />
            <input type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} />
          </label>
          <button 
            onClick={loadDocs} 
            className={`hover:text-gray-800 transition-colors ${loading ? "animate-spin" : ""}`}
            title="刷新列表"
          >
            <RefreshCcw size={14} />
          </button>
        </div>
      </div>

      {uploading && (
        <div className="text-xs text-blue-500 animate-pulse px-2 mb-2">正在上传文档...</div>
      )}

      {/* Document List */}
      <div className="flex-1 overflow-y-auto space-y-1 pr-1 font-normal custom-scrollbar">
        {docs.length === 0 && !loading && (
          <div className="text-gray-400 text-xs text-center mt-10">暂无文档，请上传</div>
        )}
        {docs.map(doc => {
          let dateStr = "未知时间";
          try {
             // ensure it's a valid date string from postgres
             dateStr = format(new Date(doc.timestamp), "yyyy-MM-dd HH:mm");
          } catch(e) {}
          
          return (
            <div 
              key={doc.id}
              className="group flex flex-col gap-1 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors relative"
            >
              <div className="flex items-center gap-2">
                <input 
                  type="checkbox" 
                  checked={selectedDocIds.includes(doc.id)}
                  onChange={() => toggleDoc(doc.id)}
                  className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                />
                <div onClick={() => toggleDoc(doc.id)} className="flex items-center gap-2 flex-1 cursor-pointer">
                  <FileText size={16} className="text-gray-400 group-hover:text-blue-500 transition-colors flex-shrink-0" />
                  <span className="truncate text-gray-800 leading-tight">{doc.name}</span>
                </div>
                
                <button 
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!confirm(`确定要彻底删除文档 "${doc.name}" 吗？该操作不可逆，且会移除知识库中的所有向量。`)) return;
                    try {
                      const { deleteDocument } = await import('@/lib/api');
                      await deleteDocument(doc.id);
                      if (selectedDocIds.includes(doc.id)) {
                        onSelectionChange(selectedDocIds.filter(v => v !== doc.id));
                      }
                      loadDocs();
                    } catch (err) {
                      alert("删除失败: " + err);
                    }
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-all ml-auto"
                  title="删除文档"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="pl-6 text-[11px] text-gray-400 flex items-center justify-between pointer-events-none">
                <span>{dateStr}</span>
                <span>{doc.chunk_count} blocks</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer / Profile */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <button className="flex items-center gap-2 px-2 py-2 w-full hover:bg-gray-100 rounded-lg transition-colors">
          <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
            <UserRound size={16} />
          </div>
          <span className="flex-1 text-left font-semibold">User Settings</span>
        </button>
      </div>
    </div>
  );
}
