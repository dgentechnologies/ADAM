import React from 'react';
import { Pause, Play, Settings as SettingsIcon, ShieldCheck } from 'lucide-react';
import logoImg from '../assets/logo.png';
import { DeviceStatus } from '../types';

interface HeaderProps {
  status: DeviceStatus;
  onTogglePause: () => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  onTogglePause,
  onOpenSettings,
}) => {
  return (
    <header className="sticky top-0 z-30 w-full glass-panel border-b border-hairline/80 px-6 py-3.5 transition-all">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-full bg-surface-container-high border border-hairline p-1 overflow-hidden shadow-inner">
            <img src={logoImg} alt="ADAM" className="w-full h-full object-contain" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-white text-base">ADAM</span>
              <span className="text-[10px] uppercase font-semibold tracking-widest px-1.5 py-0.5 rounded-full bg-white/10 text-on-surface-variant border border-white/5">
                Desktop
              </span>
            </div>
            <p className="text-[11px] text-muted tracking-tight font-medium">Companion Agent v1.0</p>
          </div>
        </div>

        {/* Center Connection Pill */}
        <div className="hidden sm:flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-near-black/80 border border-hairline shadow-subtle text-xs">
          {status.connectionState === 'connected' && !status.controlPaused && (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
              </span>
              <span className="text-white font-medium">{status.name}</span>
              <span className="text-dim">•</span>
              <span className="text-muted font-mono text-[11px]">{status.pingMs}ms</span>
            </>
          )}

          {status.controlPaused && (
            <>
              <span className="h-2 w-2 rounded-full bg-muted"></span>
              <span className="text-on-surface-variant font-medium">Laptop Control Paused</span>
            </>
          )}

          {status.connectionState === 'offline' && (
            <>
              <span className="h-2 w-2 rounded-full border border-white/40"></span>
              <span className="text-muted font-medium">ADAM Offline</span>
            </>
          )}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={onTogglePause}
            className={`px-3.5 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 border ${
              status.controlPaused
                ? 'bg-white text-black border-white hover:bg-neutral-200'
                : 'bg-surface-container-high/70 text-on-surface hover:text-white border-hairline hover:border-hairline-bright'
            }`}
            title={status.controlPaused ? 'Resume laptop control' : 'Pause laptop control'}
          >
            {status.controlPaused ? (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Resume</span>
              </>
            ) : (
              <>
                <Pause className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Pause Control</span>
              </>
            )}
          </button>

          <button
            onClick={onOpenSettings}
            className="p-2 rounded-full text-muted hover:text-white bg-surface-container-high/60 border border-hairline hover:border-hairline-bright transition-colors"
            title="Settings & Diagnostics"
          >
            <SettingsIcon className="w-4 h-4" />
          </button>

          <div className="hidden lg:flex items-center gap-1 text-[11px] text-muted pl-2 border-l border-hairline font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-white/80" />
            <span>LAN Protected</span>
          </div>
        </div>
      </div>
    </header>
  );
};
