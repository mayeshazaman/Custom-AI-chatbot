import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const Spinner = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    style={{ animation: "spin 0.8s linear infinite" }}>
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
  </svg>
);

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", gap: 8, alignItems: "flex-start",
      flexDirection: isUser ? "row-reverse" : "row",
      marginBottom: 16,
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: isUser ? "#3a3a3a" : "#e0e0e0",
        color: isUser ? "white" : "#333",
        fontSize: 11, fontWeight: 600,
      }}>
        {isUser ? "You" : "Bot"}
      </div>
      <div style={{
        maxWidth: "72%",
        background: isUser ? "#3a3a3a" : "white",
        color: isUser ? "white" : "#1a1a1a",
        borderRadius: isUser ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
        padding: "10px 14px",
        fontSize: 14,
        lineHeight: 1.6,
        border: "1px solid " + (isUser ? "#555" : "#e0e0e0"),
        whiteSpace: "pre-wrap",
      }}>
        {msg.content}
        {msg.loading && (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4, marginLeft: 6, color: "#999" }}>
            <Spinner size={12} />
          </span>
        )}
      </div>
    </div>
  );
}
// Add this after the scroll useEffect (around line 50)


export default function App() {
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [processing, setProcessing] = useState(false);
  const [sending, setSending] = useState(false);
  const [processStatus, setProcessStatus] = useState(null);
  const [indexReady, setIndexReady] = useState(false);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const sessionId = useRef(crypto.randomUUID());

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
  fetch(`${API_BASE}/health`)
    .then(res => res.json())
    .then(data => setIndexReady(data.index_ready))
    .catch(() => {}); // silently fail if backend is down
}, []);

  const addFiles = useCallback((newFiles) => {
    const pdfs = Array.from(newFiles).filter(f => f.type === "application/pdf");
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name));
      return [...prev, ...pdfs.filter(f => !names.has(f.name))];
    });
  }, []);

  const processFiles = async () => {
    if (!files.length) return;
    setProcessing(true);
    setProcessStatus(null);
    const form = new FormData();
    files.forEach(f => form.append("files", f));
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setProcessStatus({ ok: true, msg: data.message });
        setIndexReady(true);
      } else {
        setProcessStatus({ ok: false, msg: data.detail || "Processing failed." });
      }
    } catch {
      setProcessStatus({ ok: false, msg: "Cannot reach backend. Is server running?" });
    }
    setProcessing(false);
  };

  const sendMessage = async () => {
    const q = input.trim();                          // ← must be first
    if (!q || sending) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: q }]);
    setSending(true);
    const loadingId = Date.now();
    setMessages(prev => [...prev, { role: "assistant", content: "", loading: true, id: loadingId }]);
    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                question: q,
                session_id: sessionId.current    // ← add it here, inside the object
            }),
        });
        const data = await res.json();
        setMessages(prev => prev.map(m =>
            m.id === loadingId
                ? { role: "assistant", content: res.ok ? data.answer : (data.detail || "Error.") }
                : m
        ));
    } catch {
        setMessages(prev => prev.map(m =>
            m.id === loadingId ? { role: "assistant", content: "Cannot reach backend." } : m
        ));
    }
    setSending(false);
  };

  const clearConversation = async () => {
    setMessages([]);
    await fetch(`${API_BASE}/clear/${sessionId.current}`, { method: "DELETE" });
    sessionId.current = crypto.randomUUID(); // fresh session after clear
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const autoResize = (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  };
  
  
  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Arial, sans-serif", background: "#f5f5f5", color: "#1a1a1a" }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }
        textarea { resize: none; outline: none; border: none; background: transparent; font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5; color: #1a1a1a; width: 100%; }
        textarea::placeholder { color: #aaa; }
        button { cursor: pointer; border: none; outline: none; font-family: Arial, sans-serif; }
      `}</style>

      {/* Sidebar */}
      <div style={{
        width: 260, background: "#2b2b2b", color: "#f0f0f0",
        display: "flex", flexDirection: "column", padding: "20px 14px", gap: 14,
        borderRight: "1px solid #444", flexShrink: 0,
      }}>
        <div style={{ paddingBottom: 12, borderBottom: "1px solid #444" }}>
          <div style={{ fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", color: "#888", marginBottom: 4 }}>PDF Chatbot</div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>Chat with your documents</div>
        </div>

        {/* File input */}
        <div>
          <label style={{
            display: "block", border: "1.5px dashed #555", borderRadius: 8,
            padding: "16px 12px", textAlign: "center", cursor: "pointer",
            fontSize: 13, color: "#aaa", lineHeight: 1.6,
          }}>
            Click to select PDFs
            <input
              ref={fileInputRef} type="file" accept=".pdf" multiple hidden
              onChange={e => addFiles(e.target.files)}
            />
          </label>
        </div>

        {/* File list */}
        {files.length > 0 && (
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
            {files.map((f, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 8,
                background: "#3a3a3a", borderRadius: 6,
                padding: "6px 10px", fontSize: 12,
              }}>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#ccc" }}>
                  📄 {f.name}
                </span>
                <button
                  onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))}
                  style={{ background: "none", color: "#888", padding: 2, fontSize: 14 }}>
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Process button */}
        <button
          onClick={processFiles}
          disabled={!files.length || processing}
          style={{
            background: files.length && !processing ? "#f0f0f0" : "#3a3a3a",
            color: files.length && !processing ? "#1a1a1a" : "#666",
            borderRadius: 7, padding: "9px 14px",
            fontSize: 13, fontWeight: 600,
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            transition: "all 0.2s",
          }}>
          {processing ? <><Spinner size={13} /> Processing…</> : "Process & Index PDFs"}
        </button>

        {processStatus && (
          <div style={{
            fontSize: 12, padding: "8px 10px", borderRadius: 6,
            background: processStatus.ok ? "rgba(50,150,80,0.2)" : "rgba(180,50,50,0.2)",
            color: processStatus.ok ? "#7ec99a" : "#f09090",
            border: `1px solid ${processStatus.ok ? "#3a7a5a" : "#7a3030"}`,
            lineHeight: 1.5,
          }}>
            {processStatus.ok ? "✅ " : "❌ "}{processStatus.msg}
          </div>
        )}

        {/* Clear chat */}
        <button
          onClick={() => setMessages([])}
          style={{
            marginTop: "auto", background: "none", color: "#777",
            fontSize: 12, padding: "10px 0", fontFamily: "inherit",
            borderTop: "1px solid #444",
          }}>
          🗑 Clear conversation
        </button>
      </div>

      {/* Main chat */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Header */}
        <div style={{
          padding: "14px 22px", borderBottom: "1px solid #ddd",
          background: "white", display: "flex", alignItems: "center", gap: 8,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: indexReady ? "#4caf78" : "#ccc",
          }} />
          <span style={{ fontSize: 13, color: "#888" }}>
            {indexReady ? "Ready — ask anything about your PDFs" : "Upload and process PDFs to begin"}
          </span>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>
          {messages.length === 0 && (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center",
              justifyContent: "center", height: "100%", gap: 10, color: "#bbb",
            }}>
              <div style={{ fontSize: 13, fontStyle: "italic" }}>No messages yet</div>
              <div style={{ fontSize: 12, color: "#ccc" }}>Upload PDFs from the sidebar to get started</div>
            </div>
          )}
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "14px 22px", borderTop: "1px solid #ddd", background: "white" }}>
          <div style={{
            display: "flex", alignItems: "flex-end", gap: 10,
            background: "#f5f5f5", borderRadius: 12,
            border: "1.5px solid #ddd", padding: "10px 12px",
          }}>
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={e => { setInput(e.target.value); autoResize(e); }}
              onKeyDown={onKeyDown}
              placeholder={indexReady ? "Ask a question about your documents…" : "Process PDFs first to enable chat"}
              disabled={!indexReady || sending}
              style={{ flex: 1, minHeight: 24 }}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || !indexReady || sending}
              style={{
                width: 34, height: 34, borderRadius: 8, flexShrink: 0,
                background: input.trim() && indexReady && !sending ? "#2b2b2b" : "#e0e0e0",
                color: input.trim() && indexReady && !sending ? "white" : "#aaa",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 16, transition: "all 0.2s",
              }}>
              {sending ? <Spinner size={14} /> : "↑"}
            </button>
          </div>
          <div style={{ fontSize: 11, color: "#bbb", textAlign: "center", marginTop: 6 }}>
            Enter to send · Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}