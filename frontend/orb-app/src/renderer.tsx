import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import Orb from './Orb';
import './index.css';

type Message = { role: 'user' | 'ai'; text: string };
type OrbMode = 'idle' | 'querying';

function App() {
  const [mode, setMode] = useState<OrbMode>('idle');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [queryMode, setQueryMode] = useState<'auto'|'fast'|'deep'|'both'>('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const computeSize = () => {
    const maxDim = Math.max(window.innerWidth, window.innerHeight);
    const raw = 2000 / maxDim;           // 2000 is arbitrary scale factor
    return Math.min(Math.max(raw, 200), 1200); // clamp between 200 and 1200px
  };
  const [baseSize, setBaseSize] = useState(computeSize());

  // update baseSize whenever window resizes so orb scales with window size
  useEffect(() => {
    const update = () => {
      setBaseSize(computeSize());
    };
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  const userMsg = input.trim();
  setInput('');
  setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
  setMode('querying');

  try {
    const res = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: userMsg, mode: queryMode }),
    });
    const data = await res.json();

    if (data.fast && data.deep) {
      // both mode – show all fields for each
      setMessages(prev => [
        ...prev,
        { role: 'ai', text: `[FAST] ${data.fast.answer}` },
        { role: 'ai', text: `[FAST SUMMARY] ${data.fast.summary}` },
        { role: 'ai', text: `[FAST CITATIONS] ${data.fast.citations}` },
        { role: 'ai', text: `[FAST DETAILED] ${data.fast.detailed}` },
        { role: 'ai', text: `[DEEP] ${data.deep.answer}` },
        { role: 'ai', text: `[DEEP SUMMARY] ${data.deep.summary}` },
        { role: 'ai', text: `[DEEP CITATIONS] ${data.deep.citations}` },
        { role: 'ai', text: `[DEEP DETAILED] ${data.deep.detailed}` },
      ]);
    } else {
      // regular single-mode response
      setMessages(prev => [...prev, { role: 'ai', text: data.answer }]);
      if (queryMode === 'deep') {
        setMessages(prev => [
          ...prev,
          { role: 'ai', text: `[SUMMARY] ${data.summary}` },
          { role: 'ai', text: `[CITATIONS] ${data.citations}` },
          { role: 'ai', text: `[DETAILED] ${data.detailed}` },
        ]);
      }
    }
  } catch (err) {
    setMessages(prev => [...prev, { role: 'ai', text: 'Error connecting to Aion.' }]);
  }

  setMode('idle');
};


return (
  <div
    style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: '#000',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'sans-serif',
      color: '#fff',
      overflow: 'hidden',
      gap: '24px',
      padding: '20px',
    }}
    >
      {/* Top spacer (for nicer vertical balance) */}
      <div style={{ flex: 1 }} />

      {/* toolbar: ingest button + mode selector */}
      <div style={{ position: 'fixed', top: '20px', left: '20px', zIndex: 50, display: 'flex', gap: '8px' }}>
        <button
          onClick={async () => {
            try {
              const res = await fetch('http://localhost:8000/ingest', { method: 'POST' });
              const data = await res.json();
              setMessages(prev => [...prev, { role: 'ai', text: `Ingest completed. Topics: ${data.topics.join(', ')}` }]);
            } catch {
              setMessages(prev => [...prev, { role: 'ai', text: 'Ingest failed.' }]);
            }
          }}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#5533ff', color: '#fff', border: 'none' }}
          disabled={mode === 'querying'}
        >
          Ingest
        </button>
        <select
          value={queryMode}
          onChange={e => setQueryMode(e.target.value as any)}
          disabled={mode === 'querying'}
          style={{ padding: '6px', borderRadius: '6px', backgroundColor: '#111', color: '#fff', border: '1px solid #333' }}
        >
          <option value="auto">Auto</option>
          <option value="fast">Fast</option>
          <option value="deep">Deep</option>
          <option value="both">Both</option>
        </select>
      </div>

      {/* Orb + Idle label (fixed center) */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 1,                        // keep behind UI
          pointerEvents: 'none',            // clicks pass through
          backgroundColor: 'rgba(0,0,0,0.1)', // light shade to confirm visibility
        }}
      >
        <Orb
          hue={mode === 'querying' ? 200 : 0}
          hoverIntensity={mode === 'querying' ? 2 : 0.4}
          rotateOnHover={true}
          forceHoverState={mode === 'querying'}
          backgroundColor="#000000"
        />
        <div
          style={{
            position: 'absolute',
            bottom: '-32px',
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '11px',
            letterSpacing: '0.18em',
            color: mode === 'querying' ? '#7df9ff' : '#777',
            textTransform: 'uppercase',
            transition: 'color 0.4s',
          }}
        >
          {mode === 'querying' ? 'Thinking…' : 'Idle'}
        </div>
      </div>

      {/* Chat history */}
        <div
        style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 30,
            width: '80%',
            maxWidth: '960px',
            maxHeight: '40vh',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            padding: '16px',
            borderRadius: '20px',  // Softer
            // Glassmorphism for the whole box:
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(25px)',
            WebkitBackdropFilter: 'blur(25px)',
            border: '1px solid rgba(255, 255, 255, 0.18)',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5)',
        }}
        >
        {messages.map((m, i) => (
            <div
            key={i}
            style={{
                padding: '12px 16px',
                backgroundColor: m.role === 'user' ? 'rgba(128, 0, 128, 0.5)' : 'rgba(255, 255, 255, 0.12)',  // Purple for user
                backdropFilter: 'blur(15px)',
                WebkitBackdropFilter: 'blur(15px)',
                borderRadius: '18px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: 'rgba(255, 255, 255, 0.95)',
                textShadow: '0 1px 2px rgba(0,0,0,0.5)', 
                maxWidth: '85%', 
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            }}
            >
            {m.text}
            </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar fixed at bottom center */}
      <form
        onSubmit={handleSubmit}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '80%',
          maxWidth: '640px',
          display: 'flex',
          gap: '10px',
          zIndex: 20,      // ensure input sits above orb
        }}
      >
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask Aion..."
          disabled={mode === 'querying'}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: '999px',
            border: '1px solid #333',
            backgroundColor: '#111',
            color: '#fff',
            fontSize: '14px',
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={mode === 'querying' || !input.trim()}
          style={{
            padding: '12px 24px',
            borderRadius: '999px',
            border: 'none',
            backgroundColor: mode === 'querying' ? '#222' : '#5533ff',
            color: '#fff',
            cursor: mode === 'querying' ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            transition: 'background 0.3s, transform 0.1s',
          }}
        >
          {mode === 'querying' ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
