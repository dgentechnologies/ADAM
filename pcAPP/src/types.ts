export interface AdamAction {
  id: string;
  name: string;
  description: string;
  category: 'audio' | 'display' | 'system' | 'media';
  enabled: boolean;
  endpoint: string;
  icon: string;
  isDestructive?: boolean;
}

export interface ActivityLogItem {
  id: string;
  timestamp: string;
  actionId: string;
  actionName: string;
  status: 'success' | 'failed' | 'paused';
  details: string;
  latencyMs: number;
}

export interface DeviceStatus {
  name: string;
  serialNumber: string;
  ipAddress: string;
  port: number;
  connectionState: 'connected' | 'searching' | 'offline';
  pingMs: number;
  lastCommand: string;
  lastCommandTime: string;
  controlPaused: boolean;
  tokenHash: string;
}

export interface AppSettings {
  googleAccount: string;
  launchAtLogin: boolean;
  actionNotifications: boolean;
  manualIp: string;
  port: number;
  version: string;
}
