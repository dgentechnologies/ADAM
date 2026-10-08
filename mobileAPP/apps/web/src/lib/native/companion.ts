import { Capacitor, registerPlugin } from '@capacitor/core';

export interface PhoneNotification {
  id: string;
  packageName: string;
  appName: string;
  title: string;
  body: string;
  postedAt: number;
  receivedAt: number;
}
export interface NotificationAccess {
  accessGranted: boolean;
  captureEnabled: boolean;
  connected: boolean;
  error: string;
}

interface CompanionPlugin {
  notificationStatus(): Promise<NotificationAccess>;
  getNotifications(): Promise<{ items: PhoneNotification[] }>;
  setNotificationCapture(options: { enabled: boolean }): Promise<void>;
  clearNotifications(options: { reset?: boolean }): Promise<void>;
  openNotificationAccessSettings(): Promise<void>;
  setAppearance(options: { theme: 'dark' | 'light' }): Promise<void>;
  getSecret(options: { key: string }): Promise<{ value: string | null }>;
  setSecret(options: { key: string; value: string }): Promise<void>;
  clearSecrets(): Promise<void>;
  removeSecret(options: { key: string }): Promise<void>;
  openWifiSettings(): Promise<void>;
  openAppSettings(): Promise<void>;
}
export const Companion = registerPlugin<CompanionPlugin>('Companion');
export async function openWifiSettings() {
  if (!Capacitor.isNativePlatform())
    throw new Error('Open Wi-Fi settings on your phone to manage its connection.');
  await Companion.openWifiSettings();
}
