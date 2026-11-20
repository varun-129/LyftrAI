import React, { useState, useEffect, useRef } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function generateSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return 's-' + crypto.randomUUID().slice(0, 8);
  }
  return 's-' + Math.random().toString(36).substring(2, 10);
}

export default function App() {
  const [sessionId] = useState(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState('');
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [simulateFailure, setSimulateFailure] = useState('');
  const [showDevTools, setShowDevTools] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    const trimmed = inputMessage.trim();
    if (!trimmed || isLoading) return;

    const userMsg = {
      id: 'u-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6),
      role: 'user',
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const headers = {
        'Content-Type': 'application/json',
      };
      if (simulateFailure.trim()) {
        headers['X-Simulate-Failure'] = simulateFailure.trim();
      }

      const payload = {
        session_id: sessionId,
        message: trimmed,
      };
      if (systemPrompt.trim()) {
        payload.system_prompt = systemPrompt.trim();
      }

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let errorText = 'An error occurred';
        try {
          const errData = await res.json();
          if (res.status === 503) {
            errorText = 'All providers are currently unavailable, please try again';
          } else if (errData && errData.detail) {
            errorText = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          } else {
            errorText = `Error ${res.status}: ${res.statusText}`;
          }
        } catch {
          errorText = `Request failed with status ${res.status}`;
        }

        setMessages((prev) => [
          ...prev,
          {
            id: 'err-' + Date.now(),
            role: 'error',
            content: errorText,
          },
        ]);
        return;
      }

      const data = await res.json();
      const assistantMsg = {
        id: 'a-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6),
        role: 'assistant',
        content: data.reply,
        metadata: data.metadata,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: 'err-' + Date.now(),
          role: 'error',
          content: err.message || 'Network error: could not connect to server',
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-title-row">
          <h1>LLM Chatbot</h1>
          <div className="header-badges">
            <span className="session-badge">Session: {sessionId}</span>
            <button
              type="button"
              className="dev-toggle-btn"
              onClick={() => setShowDevTools((prev) => !prev)}
            >
              ⚙ Dev Tools
            </button>
          </div>
        </div>

        {showDevTools && (
          <div className="dev-tools-panel">
            <label className="dev-label">
              <span>X-Simulate-Failure header (e.g. key_1=429,key_2=500):</span>
              <input
                type="text"
                value={simulateFailure}
                onChange={(e) => setSimulateFailure(e.target.value)}
                placeholder="key_1=429"
                className="dev-input"
              />
            </label>
          </div>
        )}

        <div className="system-prompt-bar">
          <label htmlFor="system-prompt" className="system-prompt-label">
            System Prompt:
          </label>
          <input
            id="system-prompt"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional system prompt (overrides default for this session)..."
            className="system-prompt-field"
          />
        </div>
      </header>

      <main className="messages-area">
        {messages.length === 0 && !isLoading && (
          <div className="empty-state">
            <p>Ready to assist. Send a message to start the conversation.</p>
          </div>
        )}

        {messages.map((msg) => {
          if (msg.role === 'user') {
            return (
              <div key={msg.id} className="message-wrapper user-wrapper">
                <div className="user-bubble">{msg.content}</div>
              </div>
            );
          }

          if (msg.role === 'error') {
            return (
              <div key={msg.id} className="message-wrapper error-wrapper">
                <div className="error-bubble">{msg.content}</div>
              </div>
            );
          }

          if (msg.role === 'assistant') {
            const meta = msg.metadata || {};
            const isCache = meta.resolution_source === 'LLM_CACHE';

            return (
              <div
                key={msg.id}
                className="message-wrapper assistant-wrapper"
                data-testid="assistant-message"
              >
                <div className="assistant-bubble">
                  <div className="reply-text">{msg.content}</div>
                  <div className="metadata-container">
                    <div className="meta-row">
                      <span className="meta-label">Source:</span>
                      <span
                        data-testid="meta-source"
                        className={`source-badge ${isCache ? 'badge-cache' : 'badge-live'}`}
                      >
                        {meta.resolution_source}
                      </span>

                      <span className="meta-label">Model:</span>
                      <span data-testid="meta-model" className="meta-val">
                        {meta.model}
                      </span>

                      <span className="meta-label">Latency:</span>
                      <span data-testid="meta-latency" className="meta-val">
                        {meta.latency_ms} ms
                      </span>
                    </div>

                    <div className="meta-row token-row">
                      <span className="meta-label">Input Tokens:</span>
                      <span data-testid="meta-input-tokens" className="meta-val">
                        {meta.input_tokens}
                      </span>

                      <span className="meta-label">Output Tokens:</span>
                      <span data-testid="meta-output-tokens" className="meta-val">
                        {meta.output_tokens}
                      </span>

                      {meta.key_id && (
                        <>
                          <span className="meta-label muted">Key:</span>
                          <span className="meta-val muted">{meta.key_id}</span>
                        </>
                      )}

                      {meta.fallback_attempts > 0 && (
                        <>
                          <span className="meta-label muted">Fallbacks:</span>
                          <span className="meta-val muted">{meta.fallback_attempts}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          }

          return null;
        })}

        {isLoading && (
          <div className="message-wrapper assistant-wrapper">
            <div className="assistant-bubble typing-bubble">
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="chat-footer">
        <form onSubmit={handleSendMessage} className="input-form">
          <textarea
            ref={textareaRef}
            data-testid="chat-input"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
            rows={2}
            className="chat-textarea"
            disabled={isLoading}
          />
          <button
            type="submit"
            data-testid="send-button"
            className="send-btn"
            disabled={isLoading || !inputMessage.trim()}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </footer>
    </div>
  );
}
