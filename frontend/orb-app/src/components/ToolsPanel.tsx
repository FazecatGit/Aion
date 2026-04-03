import React from 'react';
import { ActiveMode, API_BASE } from '../types';

interface ToolsPanelProps {
  showToolsPanel: boolean;
  setShowToolsPanel: (fn: (prev: boolean) => boolean) => void;
  activeMode: ActiveMode;
  selectedFilePath: string | null;
  toolsLoading: boolean;
  toolsOutput: string | null;
  runTool: (endpoint: string, body: any) => void;
}

export const ToolsPanel: React.FC<ToolsPanelProps> = ({
  showToolsPanel, setShowToolsPanel, activeMode, selectedFilePath,
  toolsLoading, toolsOutput, runTool,
}) => {
  const TOOL_BTN: React.CSSProperties = {
    width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333',
    backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left',
  };

  return (
    <>
      <button
        onClick={() => setShowToolsPanel(p => !p)}
        style={{
          position: 'fixed', top: '20px', right: activeMode === 'agent' ? '320px' : '20px', zIndex: 51,
          padding: '6px 12px', borderRadius: '6px', border: '1px solid #555',
          backgroundColor: showToolsPanel ? 'rgba(85,51,255,0.3)' : '#222', color: '#fff',
          cursor: 'pointer', fontSize: '12px', transition: 'right 0.25s ease',
        }}
      >
        🔧 Tools
      </button>

      {showToolsPanel && (
        <div style={{
          position: 'fixed', top: '60px', right: '20px', zIndex: 50, width: '340px', maxHeight: '80vh',
          overflowY: 'auto', padding: '16px', borderRadius: '14px',
          backgroundColor: 'rgba(10,10,20,0.95)', border: '1px solid rgba(85,51,255,0.3)',
          backdropFilter: 'blur(20px)', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}>
          <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#5533ff', marginBottom: '12px', letterSpacing: '0.08em' }}>DEV TOOLS</div>

          {[
            { label: '🔍 Lint Current File', onClick: () => selectedFilePath && runTool('/tools/lint', { file_path: selectedFilePath }), needsFile: true },
            { label: '🔧 Lint + Auto-fix', onClick: () => selectedFilePath && runTool('/tools/lint', { file_path: selectedFilePath, fix: true }), needsFile: true },
            { label: '🏷 Type Check', onClick: () => selectedFilePath && runTool('/tools/type_check', { file_path: selectedFilePath }), needsFile: true },
            { label: '🧪 Run Pytest', onClick: () => runTool('/tools/pytest', { target: '.', with_coverage: false }) },
            { label: '📝 Git Diff (unstaged)', onClick: () => runTool('/tools/git_diff', { ref: 'HEAD' }) },
            { label: '📋 Git Diff (staged)', onClick: () => runTool('/tools/git_diff_staged', {}) },
            { label: '📜 Git Log (recent)', onClick: () => runTool('/tools/git_log', { count: 15 }) },
            { label: '✅ Pre-commit Check', onClick: () => runTool('/tools/pre_commit', {}) },
          ].map(tool => (
            <div key={tool.label} style={{ marginBottom: '10px' }}>
              <button
                onClick={tool.onClick}
                disabled={toolsLoading || (tool.needsFile && !selectedFilePath)}
                style={TOOL_BTN}
              >
                {tool.label}
              </button>
            </div>
          ))}

          {!selectedFilePath && (
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px' }}>
              Select a file in agent mode to enable file-specific tools.
            </div>
          )}

          {toolsLoading && <div style={{ color: '#5533ff', fontSize: '12px', marginTop: '8px' }}>Running...</div>}
          {toolsOutput && (
            <pre style={{
              marginTop: '10px', padding: '10px', borderRadius: '8px', backgroundColor: '#0a0a0a',
              border: '1px solid #333', color: '#ccc', fontSize: '11px', fontFamily: 'monospace',
              whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '300px', overflowY: 'auto',
            }}>
              {toolsOutput}
            </pre>
          )}
        </div>
      )}
    </>
  );
};
