'use client';
import { Capacitor } from '@capacitor/core';
import { useEffect, useState } from 'react';
import { Page, Panel, Notice } from '@/components/companion-ui';
export default function UpdatesPage() {
  const [version, setVersion] = useState('0.2.1');
  const [build, setBuild] = useState('7');
  useEffect(() => {
    if (Capacitor.isNativePlatform())
      void import('@capacitor/app')
        .then(({ App }) => App.getInfo())
        .then((info) => {
          setVersion(info.version);
          setBuild(info.build);
        });
  }, []);
  return (
    <Page title="App & updates" back="/settings">
      <Panel>
        <p className="eyebrow mb-3">INSTALLED ON THIS PHONE</p>
        <h2 className="page-title">ADAM {version}</h2>
        <p className="text-fg-muted mt-3 text-xs">Build {build} · Android companion</p>
      </Panel>
      <Panel>
        <h3 className="mb-3 text-sm font-semibold">What’s new</h3>
        <ul className="text-fg-muted space-y-3 text-sm leading-6">
          <li>Private memories with editing and search.</li>
          <li>A real gallery with camera, import, and sharing.</li>
          <li>Secure Home Assistant connections.</li>
          <li>Personal preferences, backups, and clear privacy controls.</li>
        </ul>
      </Panel>
      <Notice>
        This installation is updated with a signed APK from DGEN Technologies. An automatic update
        service is not connected.
      </Notice>
      <a
        href="https://dgentechnologies.com/products/adam"
        target="_blank"
        rel="noopener noreferrer"
        className="border-border-strong rounded-full border px-5 py-4 text-center text-sm"
      >
        Visit the official ADAM page
      </a>
      <Panel>
        <p className="eyebrow mb-3">DEMO COMPANION</p>
        <h3 className="text-sm font-medium">Firmware 1.0 · Up to date</h3>
        <p className="text-fg-muted mt-3 text-xs leading-6">
          Sample firmware status for the ADAM demo. This does not install software on a robot.
        </p>
      </Panel>
    </Page>
  );
}
