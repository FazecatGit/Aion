import React, { useRef } from 'react';

interface LightboxProps {
  url: string;
  zoom: number;
  pan: { x: number; y: number };
  setZoom: (fn: (z: number) => number) => void;
  setPan: (v: { x: number; y: number }) => void;
  onClose: () => void;
}

export const Lightbox: React.FC<LightboxProps> = ({ url, zoom, pan, setZoom, setPan, onClose }) => {
  const dragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      onWheel={(e) => {
        e.preventDefault();
        setZoom(z => Math.max(0.5, Math.min(5, z + (e.deltaY < 0 ? 0.25 : -0.25))));
      }}
      style={{
        position: 'fixed', inset: 0, zIndex: 999,
        backgroundColor: 'rgba(0,0,0,0.85)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: zoom > 1 ? 'grab' : 'zoom-in',
        overflow: 'hidden',
      }}
    >
      <img
        src={url}
        alt="Enlarged"
        draggable={false}
        onMouseDown={(e) => {
          if (zoom > 1) {
            dragging.current = true;
            dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
            e.preventDefault();
          }
        }}
        onMouseMove={(e) => {
          if (dragging.current) {
            setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
          }
        }}
        onMouseUp={() => { dragging.current = false; }}
        onMouseLeave={() => { dragging.current = false; }}
        onClick={(e) => {
          e.stopPropagation();
          if (zoom <= 1) setZoom(() => 2);
        }}
        style={{
          maxWidth: '90vw', maxHeight: '90vh',
          borderRadius: '10px',
          border: '1px solid rgba(179,136,255,0.4)',
          boxShadow: '0 0 60px rgba(0,0,0,0.9)',
          transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          transition: dragging.current ? 'none' : 'transform 0.15s',
          cursor: zoom > 1 ? (dragging.current ? 'grabbing' : 'grab') : 'zoom-in',
          userSelect: 'none',
        }}
      />
      <div style={{
        position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: '8px', alignItems: 'center', zIndex: 1000,
        backgroundColor: 'rgba(0,0,0,0.7)', borderRadius: '20px', padding: '6px 14px',
      }}>
        <button onClick={(e) => { e.stopPropagation(); setZoom(z => Math.max(0.5, z - 0.25)); }}
          style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px', padding: '2px 6px' }}>−</button>
        <span style={{ color: '#aaa', fontSize: '12px', minWidth: '40px', textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
        <button onClick={(e) => { e.stopPropagation(); setZoom(z => Math.min(5, z + 0.25)); }}
          style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px', padding: '2px 6px' }}>+</button>
        <button onClick={(e) => { e.stopPropagation(); setZoom(() => 1); setPan({ x: 0, y: 0 }); }}
          style={{ background: 'none', border: '1px solid #555', borderRadius: '6px', color: '#888', cursor: 'pointer', fontSize: '10px', padding: '3px 8px', marginLeft: '4px' }}>Reset</button>
        <button onClick={(e) => { e.stopPropagation(); onClose(); }}
          style={{ background: 'none', border: '1px solid #555', borderRadius: '6px', color: '#ff6666', cursor: 'pointer', fontSize: '10px', padding: '3px 8px' }}>Close</button>
      </div>
    </div>
  );
};
