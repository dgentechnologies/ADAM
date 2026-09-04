import React, { useState } from 'react';
import { Header } from './components/Header';
import { StatusCard } from './components/StatusCard';
import { ActionControls } from './components/ActionControls';
import { ActivityLog } from './components/ActivityLog';
import { SettingsModal } from './components/SettingsModal';
import { AdamAction, ActivityLogItem, DeviceStatus, AppSettings } from './types';
import { Shield, BellRing } from 'lucide-react';

const INITIAL_ACTIONS: AdamAction[] = [
  {
    id: 'volume_set',
    name: 'Volume Adjustment',
    description: 'Set master system audio level, mute, unmute, and step volume.',
    category: 'audio',
    enabled: true,
    endpoint: '/control?action=volume_set',
    icon: 'volume',
  },
  {
    id: 'brightness_set',
    name: 'Screen Brightness',
    description: 'Increase, decrease, or set display backlight level via WMI.',
    category: 'display',
    enabled: true,
    endpoint: '/control?action=brightness_set',
    icon: 'sun',
  },
  {
    id: 'system_lock',
    name: 'Workstation Lock',
    description: 'Lock screen when ADAM senses user departs the room.',
    category: 'system',
    enabled: true,
    endpoint: '/control?action=lock_screen',
    icon: 'lock',
    isDestructive: true,
  },
  {
    id: 'media_toggle',
    name: 'Media Playback',
    description: 'Play, pause, or skip active media (Spotify, YouTube, browser).',
    category: 'media',
    enabled: true,
    endpoint: '/control?action=media_play_pause',
    icon: 'media',
  },
  {
    id: 'screen_capture',
    name: 'Screenshot to Gallery',
    description: 'Take a snapshot and transmit to ADAM visual context stream.',
    category: 'display',
    enabled: false,
    endpoint: '/control?action=capture_screen',
    icon: 'camera',
  },
  {
    id: 'clipboard_read',
    name: 'Clipboard Assistant',
    description: 'Allow ADAM to read clipboard contents upon voice request.',
    category: 'system',
    enabled: false,
    endpoint: '/control?action=read_clipboard',
    icon: 'clipboard',
  },
];

const INITIAL_LOGS: ActivityLogItem[] = [
  {
    id: 'log-1',
    timestamp: 'Just now',
    actionId: 'volume_set',
    actionName: 'volume_set',
    status: 'success',
    details: 'Level set to 65% by voice command',
    latencyMs: 14,
  },
  {
    id: 'log-2',
    timestamp: '2 min ago',
    actionId: 'brightness_set',
    actionName: 'brightness_set',
    status: 'success',
    details: 'Backlight calibrated to 80%',
    latencyMs: 22,
  },
  {
    id: 'log-3',
    timestamp: '14 min ago',
    actionId: 'ping',
    actionName: 'heartbeat_ping',
    status: 'success',
    details: 'Routine mDNS keep-alive handshake',
    latencyMs: 4,
  },
  {
    id: 'log-4',
    timestamp: '1 hour ago',
    actionId: 'volume_mute',
    actionName: 'volume_mute',
    status: 'success',
    details: 'System muted on user prompt',
    latencyMs: 11,
  },
];

export const App: React.FC = () => {
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatus>({
    name: 'ADAM-01',
    serialNumber: 'SN-DG-9482',
    ipAddress: '192.168.1.42',
    port: 5005,
    connectionState: 'connected',
    pingMs: 5,
    lastCommand: 'volume_set(65%) — Voice triggered',
    lastCommandTime: 'Just now',
    controlPaused: false,
    tokenHash: 'd8f9•••a201',
  });

  const [actions, setActions] = useState<AdamAction[]>(INITIAL_ACTIONS);
  const [logs, setLogs] = useState<ActivityLogItem[]>(INITIAL_LOGS);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const [settings, setSettings] = useState<AppSettings>({
    googleAccount: 'alex.chen@gmail.com',
    launchAtLogin: true,
    actionNotifications: true,
    manualIp: '',
    port: 5005,
    version: 'v1.0.0-preview',
  });

  const showToast = (message: string) => {
    if (!settings.actionNotifications) return;
    setToastMessage(message);
    setTimeout(() => {
      setToastMessage(null);
    }, 2800);
  };

  const handleTogglePause = () => {
    const nextState = !deviceStatus.controlPaused;
    setDeviceStatus((prev) => ({
      ...prev,
      controlPaused: nextState,
    }));
    showToast(nextState ? 'Laptop control paused' : 'Laptop control resumed');
  };

  const handleToggleAction = (actionId: string) => {
    setActions((prev) =>
      prev.map((act) =>
        act.id === actionId ? { ...act, enabled: !act.enabled } : act
      )
    );
  };

  const handleRunTest = (actionId: string, param?: string | number) => {
    const action = actions.find((a) => a.id === actionId);
    const actionName = action?.name || actionId;
    const details = param !== undefined ? `Parameter: ${param}` : 'Manual test trigger';

    const newLogItem: ActivityLogItem = {
      id: `log-${Date.now()}`,
      timestamp: 'Just now',
      actionId,
      actionName,
      status: deviceStatus.controlPaused ? 'paused' : 'success',
      details: deviceStatus.controlPaused
        ? `${details} (simulated, control paused)`
        : `${details} (executed via test)`,
      latencyMs: Math.floor(Math.random() * 15) + 6,
    };

    setLogs((prev) => [newLogItem, ...prev.slice(0, 24)]);
    setDeviceStatus((prev) => ({
      ...prev,
      lastCommand: `${actionId}${param !== undefined ? `(${param})` : ''} — Manual test`,
      lastCommandTime: 'Just now',
    }));

    showToast(`Test: ${actionName} applied successfully`);
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleUpdateSettings = (newSettings: Partial<AppSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  };

  const handleUnpair = () => {
    if (window.confirm('Are you sure you want to unpair this laptop from ADAM?')) {
      setDeviceStatus((prev) => ({
        ...prev,
        connectionState: 'offline',
        name: 'Unpaired',
      }));
      setIsSettingsOpen(false);
      showToast('Laptop unpaired from ADAM');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background text-on-surface">
      {/* Header */}
      <Header
        status={deviceStatus}
        onTogglePause={handleTogglePause}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
        {/* Device Status Card */}
        <StatusCard status={deviceStatus} />

        {/* Action Controls & Interactive Test Strip */}
        <ActionControls
          actions={actions}
          onToggleAction={handleToggleAction}
          onRunTest={handleRunTest}
          controlPaused={deviceStatus.controlPaused}
        />

        {/* Audit & Activity Log */}
        <ActivityLog logs={logs} onClearLogs={handleClearLogs} />

        {/* Footer & Background Service Info */}
        <footer className="pt-4 pb-8 flex flex-col sm:flex-row items-center justify-between text-xs text-muted gap-2 border-t border-hairline/60">
          <div className="flex items-center gap-2">
            <Shield className="w-3.5 h-3.5 text-white/70" />
            <span>Local Area Network Only • No external cloud port opened</span>
          </div>
          <p className="font-mono text-[11px] text-dim">
            Minimizes to System Tray • Zeroconf mDNS Active
          </p>
        </footer>
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        status={deviceStatus}
        onUpdateSettings={handleUpdateSettings}
        onUnpair={handleUnpair}
      />

      {/* Simulated Desktop Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 rounded-2xl bg-charcoal/95 border border-hairline p-4 shadow-ambient flex items-center gap-3 animate-fadeIn">
          <div className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center border border-hairline">
            <BellRing className="w-3.5 h-3.5 text-white" />
          </div>
          <div>
            <p className="text-xs font-semibold text-white">ADAM Companion</p>
            <p className="text-[11px] text-muted">{toastMessage}</p>
          </div>
        </div>
      )}
    </div>
  );
};
export default App;
