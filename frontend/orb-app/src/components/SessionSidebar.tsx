import React from 'react';
import { ChatSession } from '../types';

interface SessionSidebarProps {
  showSidebar: boolean;
  setShowSidebar: (fn: (prev: boolean) => boolean) => void;
  sessions: ChatSession[];
  sessionId: string;
  editingSessionId: string | null;
  editTitle: string;
  setEditTitle: (v: string) => void;
  setEditingSessionId: (v: string | null) => void;
  handleNewChat: () => void;
  switchSession: (id: string) => void;
  handleRenameSession: (id: string, title: string) => void;
  handleDeleteSession: (id: string) => void;
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  showSidebar, setShowSidebar, sessions, sessionId,
  editingSessionId, editTitle, setEditTitle, setEditingSessionId,
  handleNewChat, switchSession, handleRenameSession, handleDeleteSession,
}) => (
  <>
    <div
      style={{
        position: 'fixed', top: 0, left: 0, bottom: 0,
        width: showSidebar ? '260px' : '0px',
        backgroundColor: 'rgba(10,10,15,0.98)',
        borderRight: showSidebar ? '1px solid rgba(85,51,255,0.3)' : 'none',
        zIndex: 80, overflowY: 'auto', overflowX: 'hidden',
        transition: 'width 0.25s ease',
        display: 'flex', flexDirection: 'column',
      }}
    >
      {showSidebar && (
        <>
          <div style={{ padding: '16px 14px 10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#5533ff', fontWeight: 'bold', fontSize: '13px', letterSpacing: '0.08em' }}>CHATS</span>
            <button onClick={handleNewChat}
              style={{ background: 'none', border: '1px dashed #5533ff55', color: '#5533ff', padding: '4px 10px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}>
              + New
            </button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 12px' }}>
            {sessions.map(s => (
              <div key={s.session_id}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '8px 10px', marginBottom: '4px', borderRadius: '8px', cursor: 'pointer',
                  backgroundColor: s.session_id === sessionId ? 'rgba(85,51,255,0.2)' : 'transparent',
                  border: s.session_id === sessionId ? '1px solid rgba(85,51,255,0.4)' : '1px solid transparent',
                }}
                onClick={() => { if (editingSessionId !== s.session_id) switchSession(s.session_id); }}
              >
                {editingSessionId === s.session_id ? (
                  <input
                    autoFocus
                    value={editTitle}
                    onChange={e => setEditTitle(e.target.value)}
                    onBlur={() => handleRenameSession(s.session_id, editTitle)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRenameSession(s.session_id, editTitle); if (e.key === 'Escape') { setEditingSessionId(null); setEditTitle(''); } }}
                    style={{ flex: 1, padding: '2px 6px', backgroundColor: '#111', border: '1px solid #555', borderRadius: '4px', color: '#fff', fontSize: '12px', outline: 'none' }}
                    onClick={e => e.stopPropagation()}
                  />
                ) : (
                  <span style={{ flex: 1, fontSize: '12px', color: s.session_id === sessionId ? '#fff' : '#aaa', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.title || 'Untitled'}
                  </span>
                )}
                <button onClick={e => { e.stopPropagation(); setEditingSessionId(s.session_id); setEditTitle(s.title || ''); }}
                  style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', fontSize: '11px', padding: '2px' }}
                  title="Rename">✏</button>
                <button onClick={e => { e.stopPropagation(); handleDeleteSession(s.session_id); }}
                  style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '11px', padding: '2px' }}
                  title="Delete">🗑</button>
              </div>
            ))}
            {sessions.length === 0 && (
              <div style={{ color: '#555', fontSize: '12px', textAlign: 'center', paddingTop: '20px' }}>No saved chats yet</div>
            )}
          </div>
        </>
      )}
    </div>
    <button
      onClick={() => setShowSidebar(prev => !prev)}
      style={{
        position: 'fixed', top: '50%', left: showSidebar ? '260px' : '0px',
        transform: 'translateY(-50%)',
        zIndex: 81, padding: '8px 4px', borderRadius: '0 8px 8px 0',
        backgroundColor: 'rgba(85,51,255,0.3)', border: 'none', color: '#fff',
        cursor: 'pointer', fontSize: '14px', transition: 'left 0.25s ease',
        backdropFilter: 'blur(10px)',
      }}
      title={showSidebar ? 'Hide sessions' : 'Show sessions'}
    >
      {showSidebar ? '◀' : '▶'}
    </button>
  </>
);
