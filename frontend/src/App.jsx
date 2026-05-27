import { useState, useRef, useEffect } from 'react'
import { Send, MessageCircle, User, Bot, Sparkles } from 'lucide-react'
import './App.css'

const API_URL = 'http://localhost:8000'

const SUGGESTED_QUESTIONS = [
  "What is the cancellation policy?",
  "How do refunds work?",
  "When is check-in and check-out?",
  "How do I become a Superhost?",
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Clear error after 4 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 4000)
      return () => clearTimeout(timer)
    }
  }, [error])

  const sendMessage = async (text) => {
    const userMessage = text || input.trim()
    if (!userMessage || isLoading) return

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setInput('')
    setIsLoading(true)
    setError(null)

    // Add empty bot message for streaming
    setMessages(prev => [...prev, { role: 'bot', content: '' }])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      })

      if (!response.ok) throw new Error('Server error')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value, { stream: true })
        const lines = text.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') break

            try {
              const parsed = JSON.parse(data)
              if (parsed.token) {
                setMessages(prev => {
                  const updated = [...prev]
                  const lastIndex = updated.length - 1
                  const lastMsg = updated[lastIndex]
                  if (lastMsg.role === 'bot') {
                    updated[lastIndex] = {
                      ...lastMsg,
                      content: lastMsg.content + parsed.token
                    }
                  }
                  return updated
                })
              }
            } catch {
              // Skip malformed chunks
            }
          }
        }
      }
    } catch (err) {
      setError('Could not connect to the server. Make sure the backend is running.')
      // Remove the empty bot message
      setMessages(prev => prev.filter((_, i) => i !== prev.length - 1))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSuggestion = (question) => {
    sendMessage(question)
  }

  return (
    <div className="chat-app">
      {/* ── Header ── */}
      <header className="chat-header">
        <div className="header-logo">
          <Sparkles />
        </div>
        <div className="header-info">
          <h1>AirBnB Support</h1>
          <p>
            <span className="status-dot" />
            Powered by Mistral AI
          </p>
        </div>
      </header>

      {/* ── Messages ── */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">
              <MessageCircle />
            </div>
            <h2>Hi there! 👋</h2>
            <p>
              I'm your AirBnB support assistant. Ask me anything about
              bookings, cancellations, refunds, hosting, and more.
            </p>
            <div className="suggested-questions">
              {SUGGESTED_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  className="suggested-btn"
                  onClick={() => handleSuggestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? <User /> : <Bot />}
                </div>
                <div className="message-content">
                  {msg.role === 'bot' && msg.content === '' && isLoading ? (
                    <div className="typing-indicator">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                    </div>
                  ) : (
                    msg.content.split('\n').map((line, j) => (
                      <p key={j}>{line}</p>
                    ))
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* ── Input ── */}
      <div className="chat-input-area">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Ask about AirBnB policies, bookings, refunds..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={() => sendMessage()}
            disabled={!input.trim() || isLoading}
            aria-label="Send message"
          >
            <Send />
          </button>
        </div>
        <p className="input-hint">
          Responses powered by local Mistral AI · Your data stays on your device
        </p>
      </div>

      {/* ── Error Toast ── */}
      {error && <div className="error-toast">{error}</div>}
    </div>
  )
}

export default App
