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
  const [lastUserQuery, setLastUserQuery] = useState<string | null>(null);
  const computeSize = () => {
    const maxDim = Math.max(window.innerWidth, window.innerHeight);
    const raw = 2000 / maxDim;           // 2000 is arbitrary scale factor
    return Math.min(Math.max(raw, 200), 1200); // clamp between 200 and 1200px
  };
  const [baseSize, setBaseSize] = useState(computeSize());
  const [detailsContent, setDetailsContent] = useState<string | null>(null);

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

    // store last user query for potential dislike feedback
    setLastUserQuery(userMsg);

    if (data.fast && data.deep) {
      // both mode – show concise answers + summaries, stash full details
      setMessages(prev => [
        ...prev,
        { role: 'ai', text: `[FAST] ${data.fast.answer}` },
        { role: 'ai', text: `[FAST SUMMARY] ${data.fast.summary}` },
        { role: 'ai', text: `[DEEP] ${data.deep.answer}` },
        { role: 'ai', text: `[DEEP SUMMARY] ${data.deep.summary}` },
      ]);
      setDetailsContent(`FAST CITATIONS:\n${data.fast.citations}\n\nFAST DETAILED:\n${data.fast.detailed}\n\nDEEP CITATIONS:\n${data.deep.citations}\n\nDEEP DETAILED:\n${data.deep.detailed}`);
    } else {
      // regular single-mode response: show answer + summary for deep mode
      setMessages(prev => [...prev, { role: 'ai', text: data.answer }]);
      if (queryMode === 'deep') {
        setMessages(prev => [...prev, { role: 'ai', text: `[SUMMARY] ${data.summary}` }]);
        setDetailsContent(`CITATIONS:\n${data.citations}\n\nDETAILED:\n${data.detailed}`);
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

      {/* toolbar: upload + mode selector */}
      <div style={{ position: 'fixed', top: '20px', left: '20px', zIndex: 50, display: 'flex', gap: '8px' }}>
        <button
          onClick={async () => {
            try {
              const res = await fetch('http://localhost:8000/open_data_folder', { method: 'POST' });
              const json = await res.json();
              if (json.status === 'opened') {
                setMessages(prev => [...prev, { role: 'ai', text: `Opened data folder: ${json.path}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Open failed: ${json.error}` }]);
              }
            } catch (e) {
              setMessages(prev => [...prev, { role: 'ai', text: 'Open data folder failed.' }]);
            }
          }}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#333', color: '#fff', border: 'none' }}
          disabled={mode === 'querying'}
        >
          Open Data Folder
        </button>

        <input id="pdf-upload" type="file" accept="application/pdf" style={{ display: 'none' }} onChange={async (e) => {
          const f = (e.target as HTMLInputElement).files?.[0];
          if (!f) return;
          setMode('querying');
          try {
            const anyF = f as any;
            if (anyF.path) {
              const res = await fetch('http://localhost:8000/ingest_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: anyF.path })
              });
              const j = await res.json();
              if (j.status === 'ingested') {
                setMessages(prev => [...prev, { role: 'ai', text: `Ingested ${j.filename || f.name}. Topics: ${j.topics?.join(', ') || ''}` }]);
              } else if (j.status === 'exists') {
                setMessages(prev => [...prev, { role: 'ai', text: `File already ingested: ${f.name}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Ingest failed: ${j.error || JSON.stringify(j)}` }]);
              }
            } else {
              const form = new FormData();
              form.append('file', f, f.name);
              const res = await fetch('http://localhost:8000/upload_and_ingest', { method: 'POST', body: form });
              const j = await res.json();
              if (j.status === 'ingested') {
                setMessages(prev => [...prev, { role: 'ai', text: `Uploaded and ingested ${j.filename}. Topics: ${j.topics?.join(', ') || ''}` }]);
              } else if (j.status === 'exists') {
                setMessages(prev => [...prev, { role: 'ai', text: `File already exists on server: ${j.filename}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Upload failed: ${j.error || JSON.stringify(j)}` }]);
              }
            }
          } catch (err) {
            setMessages(prev => [...prev, { role: 'ai', text: 'Upload/ingest failed.' }]);
          }
          setMode('idle');
          (document.getElementById('pdf-upload') as HTMLInputElement).value = '';
        }} />

        <button
          onClick={() => (document.getElementById('pdf-upload') as HTMLInputElement).click()}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#228822', color: '#fff', border: 'none' }}
          disabled={mode === 'querying'}
        >
          Upload PDF
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
            <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, wordBreak: 'break-word' }}>{m.text}</div>
            {m.role === 'ai' && i === messages.length - 1 && lastUserQuery && (
              <div style={{ marginTop: 6 }}>
                <button
                  onClick={async () => {
                    if (!lastUserQuery) return;
                    setMode('querying');
                    try {
                      const res = await fetch('http://localhost:8000/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: lastUserQuery, prev_mode: queryMode })
                      });
                      const deep = await res.json();
                      // append deep answer and summary
                      setMessages(prev => [
                        ...prev,
                        { role: 'ai', text: `[DEEP RETRY] ${deep.answer}` },
                        { role: 'ai', text: `[SUMMARY] ${deep.summary}` }
                      ]);
                      // stash detailed for Show Details
                      setDetailsContent(`CITATIONS:\n${deep.citations}\n\nDETAILED:\n${deep.detailed}`);
                      setLastUserQuery(null);
                    } catch (e) {
                      setMessages(prev => [...prev, { role: 'ai', text: 'Feedback failed.' }]);
                    }
                    setMode('idle');
                  }}
                  style={{
                    marginLeft: 8,
                    padding: '6px 8px',
                    borderRadius: 8,
                    backgroundColor: '#aa2222',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  Dislike — Ask Deep
                </button>
              </div>
            )}
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
        {detailsContent && (
          <button
            type="button"
            onClick={() => {
              setMessages(prev => [...prev, { role: 'ai', text: detailsContent }]);
              setDetailsContent(null);
            }}
            disabled={mode === 'querying'}
            style={{
              marginRight: '8px',
              padding: '10px 12px',
              borderRadius: '12px',
              border: 'none',
              backgroundColor: '#444',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            Show Details
          </button>
        )}
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
