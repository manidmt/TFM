import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import type { ChatMessage, ChatSSEEvent } from '../api/types';
import './ChatDrawer.css';

const STORAGE_KEY = 'qr_chat_history';
const MAX_MESSAGES = 50;
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

function loadHistory(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return parsed.messages ?? [];
  } catch {
    return [];
  }
}

function saveHistory(messages: ChatMessage[]) {
  const trimmed = messages.slice(-MAX_MESSAGES);
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ messages: trimmed, updatedAt: new Date().toISOString() }),
  );
}

/** Lightweight markdown → HTML for assistant messages */
function renderMarkdown(text: string): string {
  const esc = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Inline formatting on an already-escaped string
  const inline = (s: string) =>
    s
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>');

  const blocks = text.split(/\n\n+/);
  const out: string[] = [];

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    // Fenced code block
    if (trimmed.startsWith('```')) {
      const lines = trimmed.split('\n');
      const code = esc(lines.slice(1, lines[lines.length - 1] === '```' ? -1 : undefined).join('\n'));
      out.push(`<pre><code>${code}</code></pre>`);
      continue;
    }

    // Header (## or ###)
    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)/);
    if (hMatch) {
      const level = hMatch[1].length;
      out.push(`<h${level}>${inline(esc(hMatch[2]))}</h${level}>`);
      continue;
    }

    // Unordered list block
    const lines = trimmed.split('\n');
    if (lines.every((l) => /^\s*[-*]\s/.test(l))) {
      out.push('<ul>' + lines.map((l) => `<li>${inline(esc(l.replace(/^\s*[-*]\s+/, '')))}</li>`).join('') + '</ul>');
      continue;
    }

    // Ordered list block
    if (lines.every((l) => /^\s*\d+[.)]\s/.test(l))) {
      out.push('<ol>' + lines.map((l) => `<li>${inline(esc(l.replace(/^\s*\d+[.)]\s+/, '')))}</li>`).join('') + '</ol>');
      continue;
    }

    // Regular paragraph — preserve single line breaks as <br>
    out.push('<p>' + lines.map((l) => inline(esc(l))).join('<br>') + '</p>');
  }

  return out.join('');
}

export default function ChatDrawer() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(loadHistory);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [toolActivity, setToolActivity] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, toolActivity]);

  // Save to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) saveHistory(messages);
  }, [messages]);

  if (!user || unavailable) return null;

  async function handleSend() {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput('');
    setStreaming(true);
    setToolActivity(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/api/private/chat`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
        signal: controller.signal,
      });

      if (res.status === 503) {
        setUnavailable(true);
        setMessages((prev) => prev.slice(0, -1)); // Remove the user message we just added
        return;
      }
      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6);
          let event: ChatSSEEvent;
          try {
            event = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (event.type === 'token' && event.content) {
            assistantContent += event.content;
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { role: 'assistant', content: assistantContent }];
              }
              return [...prev, { role: 'assistant', content: assistantContent }];
            });
          } else if (event.type === 'tool') {
            if (event.status === 'start') {
              setToolActivity(event.name ?? null);
            } else {
              setToolActivity(null);
            }
          } else if (event.type === 'done') {
            // Final state
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ]);
    } finally {
      setStreaming(false);
      setToolActivity(null);
      abortRef.current = null;
    }
  }

  function handleNewChat() {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  if (!open) {
    return (
      <button className="chat-fab" onClick={() => setOpen(true)} aria-label="Open chat">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </button>
    );
  }

  return (
    <>
      <div className="chat-overlay" onClick={() => setOpen(false)} />
      <div className="chat-drawer">
        <div className="chat-header">
          <span className="chat-header-title">Risk Analyst</span>
          <div className="chat-header-actions">
            <button className="chat-header-btn" onClick={handleNewChat}>New chat</button>
            <button className="chat-close-btn" onClick={() => setOpen(false)}>×</button>
          </div>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="chat-empty">
              Ask me about volatility predictions, portfolio analysis, or asset risk.
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`chat-msg ${msg.role}`}
              dangerouslySetInnerHTML={
                msg.role === 'assistant'
                  ? { __html: renderMarkdown(msg.content) }
                  : undefined
              }
            >
              {msg.role === 'user' ? msg.content : undefined}
            </div>
          ))}
          {toolActivity && (
            <div className="chat-tool-pill">Calling {toolActivity}...</div>
          )}
          {streaming && !toolActivity && messages[messages.length - 1]?.role !== 'assistant' && (
            <div className="chat-thinking">Thinking...</div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <textarea
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your portfolio..."
            rows={1}
            disabled={streaming}
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={streaming || !input.trim()}
          >
            Send
          </button>
        </div>
        <div className="chat-disclaimer">
          This is an academic research tool, not financial advice.
        </div>
      </div>
    </>
  );
}
