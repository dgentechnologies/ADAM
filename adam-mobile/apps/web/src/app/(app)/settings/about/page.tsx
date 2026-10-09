'use client';
import { AdamFaceMark } from '@adam/ui';
import { Page, Panel, Row } from '@/components/companion-ui';
import { FileText, Shield } from 'lucide-react';
export default function AboutPage() {
  return (
    <Page title="About ADAM" back="/settings">
      <div className="flex flex-col items-center gap-6 py-8">
        <AdamFaceMark size="lg" expression="happy" />
        <div className="text-center">
          <h2 className="text-2xl tracking-widest">ADAM</h2>
          <p className="text-fg-muted mt-3 text-sm">Autonomous Desktop AI Module</p>
          <p className="text-fg-muted mt-2 text-xs">Companion 0.2.2 · Made in India</p>
        </div>
      </div>
      <Panel>
        <p className="text-fg-muted text-sm leading-7">
          A more human way to connect with your everyday world. Designed and built by DGEN
          Technologies Pvt. Ltd., Kolkata, India.
        </p>
      </Panel>
      <Panel className="flush">
        <Row href="/privacy" icon={Shield} title="Privacy notice" />
        <Row href="/terms" icon={FileText} title="Terms of use" />
      </Panel>
      <a
        href="https://dgentechnologies.com"
        target="_blank"
        rel="noopener noreferrer"
        className="py-4 text-center text-sm underline underline-offset-4"
      >
        dgentechnologies.com
      </a>
    </Page>
  );
}
