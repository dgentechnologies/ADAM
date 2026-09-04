import React from 'react';
import { Radio, Wifi, KeyRound, Clock, Activity, Cpu } from 'lucide-react';
import { DeviceStatus } from '../types';

interface StatusCardProps {
  status: DeviceStatus;
}

export const StatusCard: React.FC<StatusCardProps> = ({ status }) => {
  return (
    <div className="rounded-card bg-charcoal/90 border border-hairline p-6 shadow-subtle relative overflow-hidden">
      {/* Subtle top ambient gradient */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent" />

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
        {/* Left: Device Identity & Status */}
        <div className="md:col-span-7 flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase font-semibold tracking-wider text-muted flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-white animate-pulse" />
              Connected Companion Hub
            </span>
          </div>

          <div>
            <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              {status.name}
              <span className="text-xs font-mono font-normal px-2.5 py-0.5 rounded-full bg-white/5 border border-hairline text-on-surface-variant">
                {status.serialNumber}
              </span>
            </h2>
            <p className="text-xs text-on-surface-variant mt-1 flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Wifi className="w-3.5 h-3.5 text-muted" />
                LAN: <span className="font-mono text-white">{status.ipAddress}:{status.port}</span>
              </span>
              <span className="text-dim">•</span>
              <span className="flex items-center gap-1">
                <Activity className="w-3.5 h-3.5 text-muted" />
                Ping: <span className="font-mono text-white">{status.pingMs}ms</span>
              </span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-near-black border border-hairline text-[11px] text-on-surface-variant font-mono">
              <KeyRound className="w-3 h-3 text-white/70" />
              <span>Token: {status.tokenHash}</span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-near-black border border-hairline text-[11px] text-on-surface-variant">
              <Cpu className="w-3 h-3 text-white/70" />
              <span>mDNS: <span className="text-white">_adam-laptop._tcp</span></span>
            </div>
          </div>
        </div>

        {/* Right: Last Command Received Box */}
        <div className="md:col-span-5 rounded-2xl bg-near-black/70 border border-hairline/80 p-4 flex flex-col justify-between h-full">
          <div className="flex items-center justify-between text-xs text-muted mb-2">
            <span className="uppercase tracking-wider font-semibold text-[11px] flex items-center gap-1.5">
              <Clock className="w-3 h-3" />
              Last Command Received
            </span>
            <span className="text-[11px] text-on-surface-variant font-mono">{status.lastCommandTime}</span>
          </div>

          <div className="bg-surface-container-low/90 rounded-xl px-3.5 py-3 border border-hairline/60">
            <p className="text-sm font-medium text-white tracking-tight flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
              {status.lastCommand}
            </p>
            <p className="text-[11px] text-muted mt-1">
              {status.controlPaused
                ? 'Laptop control is paused — actions won\'t be executed'
                : 'Verified via secure handshake & executed locally'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
