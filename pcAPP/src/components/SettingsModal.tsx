import React from 'react';
import {
  X,
  User,
  Power,
  Bell,
  Network,
  ShieldCheck,
  RefreshCw,
  ExternalLink,
  Unlink,
  Check
} from 'lucide-react';
import { AppSettings, DeviceStatus } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  status: DeviceStatus;
  onUpdateSettings: (newSettings: Partial<AppSettings>) => void;
  onUnpair: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  status,
  onUpdateSettings,
  onUnpair,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md animate-fadeIn">
      <div className="w-full max-w-xl rounded-card bg-charcoal border border-hairline p-6 shadow-ambient max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-hairline/80">
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-bold text-white tracking-tight">Settings & Preferences</h2>
            <span className="text-[11px] font-mono text-muted bg-white/5 px-2 py-0.5 rounded-full border border-hairline">
              {settings.version}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-muted hover:text-white hover:bg-surface-container-high transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-6 pt-5">
          {/* Account Section (§3.8) */}
          <div className="space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-white uppercase tracking-wider">
              <User className="w-3.5 h-3.5 text-muted" />
              <span>DGEN Identity & Account</span>
            </div>
            <div className="rounded-2xl bg-near-black/70 border border-hairline p-3.5 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-white">{settings.googleAccount}</p>
                <p className="text-[11px] text-muted">Linked with mobile app & cloud telemetry</p>
              </div>
              <button
                onClick={() => alert('Sign out simulation: In production, clears local auth state.')}
                className="btn-secondary text-xs py-1 px-3"
              >
                Sign Out
              </button>
            </div>
          </div>

          {/* Connected Device & Unpair (§3.8) */}
          <div className="space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-white uppercase tracking-wider">
              <Network className="w-3.5 h-3.5 text-muted" />
              <span>Linked Hardware</span>
            </div>
            <div className="rounded-2xl bg-near-black/70 border border-hairline p-3.5 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-white">{status.name} ({status.serialNumber})</p>
                <p className="text-[11px] text-muted font-mono">LAN: {status.ipAddress}:{status.port}</p>
              </div>
              <button
                onClick={onUnpair}
                className="px-3 py-1 rounded-full text-xs font-medium border border-neutral-700 bg-red-950/20 text-neutral-300 hover:text-white hover:bg-red-900/40 transition-all flex items-center gap-1.5"
              >
                <Unlink className="w-3 h-3" />
                <span>Unpair</span>
              </button>
            </div>
          </div>

          {/* Startup & Background Behavior (§3.8) */}
          <div className="space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-white uppercase tracking-wider">
              <Power className="w-3.5 h-3.5 text-muted" />
              <span>System & Startup</span>
            </div>
            <div className="space-y-2">
              <div className="rounded-2xl bg-near-black/70 border border-hairline p-3.5 flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold text-white">Launch at login</p>
                  <p className="text-[11px] text-muted">Keep agent running in the system tray</p>
                </div>
                <button
                  type="button"
                  onClick={() => onUpdateSettings({ launchAtLogin: !settings.launchAtLogin })}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors border border-hairline ${
                    settings.launchAtLogin ? 'bg-hairline-bright' : 'bg-surface-container-low'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                      settings.launchAtLogin ? 'translate-x-[18px]' : 'translate-x-[2px]'
                    } mt-[1px]`}
                  />
                </button>
              </div>

              <div className="rounded-2xl bg-near-black/70 border border-hairline p-3.5 flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold text-white flex items-center gap-1.5">
                    <Bell className="w-3.5 h-3.5 text-muted" />
                    Action Notification Toasts
                  </p>
                  <p className="text-[11px] text-muted">Show desktop banner when ADAM changes volume or brightness</p>
                </div>
                <button
                  type="button"
                  onClick={() => onUpdateSettings({ actionNotifications: !settings.actionNotifications })}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors border border-hairline ${
                    settings.actionNotifications ? 'bg-hairline-bright' : 'bg-surface-container-low'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                      settings.actionNotifications ? 'translate-x-[18px]' : 'translate-x-[2px]'
                    } mt-[1px]`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* OS Permissions Check (§3.5) */}
          <div className="space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-white uppercase tracking-wider">
              <ShieldCheck className="w-3.5 h-3.5 text-muted" />
              <span>OS Permissions & Access</span>
            </div>
            <div className="rounded-2xl bg-near-black/70 border border-hairline p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-white">Windows Audio Core (pycaw)</span>
                <span className="flex items-center gap-1 text-white text-[11px]">
                  <Check className="w-3 h-3 text-white" /> Granted
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-white">Display WMI / VCP Controls</span>
                <span className="flex items-center gap-1 text-white text-[11px]">
                  <Check className="w-3 h-3 text-white" /> Granted
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-white">Local LAN Bind Socket (:5005)</span>
                <span className="flex items-center gap-1 text-white text-[11px]">
                  <Check className="w-3 h-3 text-white" /> Active
                </span>
              </div>
            </div>
          </div>

          {/* About & Support (§3.8) */}
          <div className="pt-3 border-t border-hairline/80 flex items-center justify-between text-xs text-muted">
            <span>DGEN Technologies Pvt. Ltd.</span>
            <div className="flex items-center gap-3">
              <button
                onClick={() => alert('Checking for updates: App is on latest preview build (v1.0.0).')}
                className="hover:text-white flex items-center gap-1"
              >
                <RefreshCw className="w-3 h-3" /> Check for updates
              </button>
              <a
                href="https://dgentechnologies.com"
                target="_blank"
                rel="noreferrer"
                className="hover:text-white flex items-center gap-1"
              >
                Support <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
