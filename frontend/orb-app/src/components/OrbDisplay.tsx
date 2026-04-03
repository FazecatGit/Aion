import React from 'react';
import Orb from '../Orb';
import { OrbMode } from '../types';

interface OrbDisplayProps {
  mode: OrbMode;
}

export const OrbDisplay: React.FC<OrbDisplayProps> = ({ mode }) => (
  <div
    style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      zIndex: 1,
      pointerEvents: 'none',
      backgroundColor: 'rgba(0,0,0,0.1)',
    }}
  >
    <Orb
      hue={
        mode === 'agent-processing' ? 270 :
        mode === 'querying' ? 200 :
        0
      }
      hoverIntensity={
        mode === 'agent-processing' ? 2.5 :
        mode === 'querying' ? 2 : 0.4
      }
      rotateOnHover={true}
      forceHoverState={mode !== 'idle'}
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
        color:
          mode === 'agent-processing' ? '#ff00ff' :
          mode === 'querying' ? '#7df9ff' :
          '#777',
        textTransform: 'uppercase',
        transition: 'color 0.4s',
      }}
    >
      {mode === 'agent-processing' ? 'Processing Code\u2026' : mode === 'querying' ? 'Thinking\u2026' : 'Idle'}
    </div>
  </div>
);
