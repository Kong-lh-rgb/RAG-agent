export const API_BASE_URL = "http://localhost:8000";

import { fetchEventSource } from "@microsoft/fetch-event-source";

export interface DocumentItem {
  id: string;
  name: string;
  timestamp: string;
  chunk_count: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const res = await fetch(`${API_BASE_URL}/documents`);
  if (!res.ok) {
    throw new Error("Failed to fetch documents");
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  
  const res = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Upload failed");
  }
  return res.json();
}

export async function deleteDocument(doc_id: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/documents/${doc_id}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Delete failed");
  }
  return res.json();
}

export async function streamChat(
  query: string, 
  session_id: string, 
  doc_ids: string[], 
  onMessage: (content: string) => void,
  onDone: () => void,
  onError: (err: any) => void
) {
  const ctrl = new AbortController();
  
  try {
    await fetchEventSource(`${API_BASE_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        session_id,
        doc_ids,
        top_k: 3 // Default top_k or make configurable
      }),
      signal: ctrl.signal,
      onmessage(event) {
        if (event.event === "token") {
          try {
            const data = JSON.parse(event.data);
            if (data.content) {
              onMessage(data.content);
            }
          } catch (e) {
            console.error("Parse error on token", e);
          }
        } else if (event.event === "done") {
          onDone();
          ctrl.abort();
        }
        // "context" event could be used if we want to show references
      },
      onerror(err) {
        onError(err);
        throw err; // Stop retrying
      }
    });
  } catch (err) {
    // Error handled inside onerror or here if fetch fails
  }
}
