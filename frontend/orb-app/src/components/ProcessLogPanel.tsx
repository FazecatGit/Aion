import React, { useRef, useEffect } from 'react';

interface ProcessLog {
  ts: number;
  level: string;
  logger: string;
  message: string;
}

interface ProcessLogPanelProps {
  logs: ProcessLog[];
  expanded: boolean;
  setExpanded: (fn: (prev: boolean) => boolean) => void;
  clearLogs: () => void;
  onClose: () => void;
}

export const ProcessLogPanel: React.FC<ProcessLogPanelProps> = ({
  logs, expanded, setExpanded, clearLogs, onClose,
}) => {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs.length]);

  return (
    <div style={{
      position: 'fixed', bottom: '80px', right: '16px', zIndex: 55,
      width: expanded ? '700px' : '500px',
      maxWidth: expanded ? '70vw' : '45vw',
      height: expanded ? '550px' : '300px',
      borderRadius: '12px', overflow: 'hidden',
      backgroundColor: 'rgba(0, 5, 10, 0.92)', backdropFilter: 'blur(20px)',
      border: '1px solid rgba(255,153,0,0.2)', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      display: 'flex', flexDirection: 'column',
      transition: 'width 0.25s, height 0.25s, max-width 0.25s',
    }}>
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid #222',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#ff9900' }}>⚙ Backend Process Logs</span>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <button onClick={() => setExpanded(p => !p)}
            title={expanded ? 'Shrink panel' : 'Expand panel'}
            style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#888', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
            {expanded ? '⊟' : '⊞'}
          </button>
          <button onClick={clearLogs}
            style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#666', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
            Clear
          </button>
          <button onClick={onClose}
            style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#666', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
            ✕
          </button>
        </div>
      </div>
      <div style={{
        flex: 1, overflowY: 'auto', padding: '6px 10px',
        fontFamily: "'Fira Code', 'JetBrains Mono', Consolas, monospace",
        fontSize: expanded ? '12px' : '11px', lineHeight: '1.5',
      }}>
        {logs.length === 0 && (
          <div style={{ color: '#555', textAlign: 'center', paddingTop: '40px' }}>
            Waiting for backend activity...
          </div>
        )}
        {logs.map((log, i) => {
          const levelColor = log.level === 'ERROR' ? '#ff4444'
            : log.level === 'WARNING' ? '#ffaa33'
            : log.level === 'INFO' ? '#00cc88'
            : '#555';
          return (
            <div key={i} style={{ padding: '2px 0', borderBottom: '1px solid #111', wordBreak: 'break-word' }}>
              <span style={{ color: '#555', marginRight: '6px' }}>
                {new Date(log.ts * 1000).toLocaleTimeString()}
              </span>
              <span style={{ color: levelColor, fontWeight: 'bold', marginRight: '6px' }}>
                {log.level}
              </span>
              <span style={{ color: '#7d7d7d', marginRight: '6px' }}>
                [{log.logger}]
              </span>
              <span style={{ color: '#ccc' }}>{log.message}</span>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
};
