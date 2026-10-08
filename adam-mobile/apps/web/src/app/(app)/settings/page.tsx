'use client';
import { Toggle } from '@adam/ui';
import {
  Brain,
  Bluetooth,
  Cpu,
  Database,
  Info,
  Laptop,
  Mic,
  Shield,
  Sun,
  UserRound,
  Wifi,
  House,
  Bell,
} from 'lucide-react';
import { Page, Panel, Row } from '@/components/companion-ui';
import { useAppStore } from '@/stores/app-store';
import { useLocalData } from '@/lib/use-local-data';
import { useDemoDevice } from '@/lib/demo-device';
export default function SettingsPage() {
  const { data } = useLocalData();
  const { device } = useDemoDevice();
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  return (
    <Page title="Settings">
      <div>
        <p className="eyebrow mb-3">MAKE IT FEEL LIKE YOU</p>
        <h2 className="page-title">A few personal touches.</h2>
      </div>
      <Panel className="flush">
        <Row
          href="/settings/account"
          icon={UserRound}
          title={data.name || 'Your profile'}
          detail="Account & personal details"
        />
      </Panel>
      <div>
        <p className="eyebrow mb-3">YOUR PHONE</p>
        <Panel className="flush">
          <Row
            href="/notifications"
            icon={Bell}
            title="Notifications & background"
            detail="Permission, reading & your private inbox"
          />
          <Row
            href="/settings/privacy"
            icon={Database}
            title="Your data"
            detail="Backup, restore & erase local data"
          />
        </Panel>
        <Panel className="mt-3">
          <div className="flex items-center gap-3">
            <Sun size={20} />
            <Toggle
              label="Light appearance"
              checked={theme === 'light'}
              onCheckedChange={(value) => setTheme(value ? 'light' : 'dark')}
            />
          </div>
        </Panel>
      </div>
      <div>
        <p className="eyebrow mb-3">CONNECTIONS</p>
        <Panel className="flush">
          <Row
            href="/smart-home"
            icon={House}
            title="Smart home"
            detail="Home Assistant connection"
          />
          <Row
            href="/settings/wifi"
            icon={Wifi}
            title="Wi-Fi"
            detail="Your phone’s network settings"
          />
        </Panel>
      </div>
      <div>
        <p className="eyebrow mb-3">FOR YOUR ADAM</p>
        <Panel className="flush">
          <Row
            href={device.connected ? '/device' : '/discover'}
            icon={Bluetooth}
            title={device.connected ? 'Your ADAM' : 'Connect ADAM'}
            detail={
              device.connected
                ? `${device.name} · Demo connected`
                : 'Explore pairing & your demo companion'
            }
          />
          <Row
            href="/settings/ai-brain"
            icon={Brain}
            title="AI preferences"
            detail="Model access & your own key"
          />
          <Row
            href="/settings/voice"
            icon={Mic}
            title="Voice & wake word"
            detail="Save your preferences for ADAM"
          />
          <Row
            href="/settings/laptops"
            icon={Laptop}
            title="Laptop companion"
            detail="Explore your connected desk in the demo"
          />
        </Panel>
      </div>
      <div>
        <p className="eyebrow mb-3">ABOUT & PRIVACY</p>
        <Panel className="flush">
          <Row
            href="/privacy"
            icon={Shield}
            title="Privacy"
            detail="How your information is handled"
          />
          <Row
            href="/settings/software-update"
            icon={Cpu}
            title="App version & updates"
            detail="ADAM Companion 0.2.1"
          />
          <Row
            href="/settings/about"
            icon={Info}
            title="About ADAM"
            detail="Made by DGEN Technologies"
          />
        </Panel>
      </div>
      <p className="text-fg-muted text-center text-[11px]">ADAM Companion · 0.2.1</p>
    </Page>
  );
}
