'use client';
import { AdamFaceMark, buttonVariants } from '@adam/ui';
import {
  ArrowUpRight,
  Bell,
  Bluetooth,
  Brain,
  Images,
  LampCeiling,
  Plus,
  UserRound,
} from 'lucide-react';
import Link from 'next/link';
import { Page, Panel, Notice } from '@/components/companion-ui';
import { useLocalData } from '@/lib/use-local-data';
import { useMemoryIntent } from '@/stores/memory-intent';
import { useDemoDevice } from '@/lib/demo-device';
export default function HomePage() {
  const { data, error } = useLocalData();
  const requestMemory = useMemoryIntent((state) => state.request);
  const { device } = useDemoDevice();
  return (
    <Page
      title="ADAM"
      action={
        <div className="flex items-center gap-2">
          <Link
            href="/notifications"
            aria-label="Notifications"
            className="border-border flex h-11 w-11 items-center justify-center rounded-full border"
          >
            <Bell size={19} />
          </Link>
          <Link
            href="/settings/account"
            aria-label="Your profile"
            className="border-border flex h-11 w-11 items-center justify-center rounded-full border"
          >
            <UserRound size={19} />
          </Link>
        </div>
      }
    >
      <div>
        <p className="eyebrow mb-3">YOUR EVERYDAY COMPANION</p>
        <h2 className="page-title break-words">
          {data.name ? `Hello, ${data.name.split(' ')[0]}.` : 'Make yourself at home.'}
        </h2>
      </div>
      {error && <Notice error>{error}</Notice>}
      <section className="hero-halo -mx-3 flex flex-col items-center px-3 pb-6 text-center">
        <div className="flex h-36 items-center justify-center">
          <AdamFaceMark expression={device.connected ? device.expression : 'idle'} size="xl" />
        </div>
        <span className="border-border text-fg-muted rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[.14em]">
          {device.connected ? 'Demo companion connected' : 'Ready when you are'}
        </span>
        <p className="text-fg-muted mt-4 max-w-xs break-words text-sm leading-6">
          Your memories and moments live here.
          <br />
          {device.connected
            ? `${device.name} is here to explore with you.`
            : 'Meet your companion in the ADAM demo.'}
        </p>
        <Link
          href={device.connected ? '/device' : '/discover'}
          className="mt-4 flex min-h-11 items-center gap-2 text-sm"
        >
          <Bluetooth size={16} />
          {device.connected ? 'Open your ADAM' : 'Meet your ADAM'}
          <ArrowUpRight size={15} />
        </Link>
      </section>
      <div className="grid grid-cols-2 gap-3">
        {[
          {
            href: '/memory',
            Icon: Brain,
            title: 'Memory',
            text: data.facts.length
              ? `${data.facts.length} saved ${data.facts.length === 1 ? 'memory' : 'memories'}`
              : 'The things that matter',
          },
          { href: '/gallery', Icon: Images, title: 'Moments', text: 'Your personal gallery' },
        ].map(({ href, Icon, title, text }) => (
          <Link key={href} href={href} className="panel">
            <div className="mb-7 flex items-center justify-between">
              <Icon size={24} strokeWidth={1.3} />
              <ArrowUpRight size={16} className="text-fg-muted" />
            </div>
            <h3 className="text-base font-medium">{title}</h3>
            <p className="text-fg-muted mt-2 text-xs">{text}</p>
          </Link>
        ))}
      </div>
      <Link href="/smart-home">
        <Panel>
          <div className="flex items-center gap-4">
            <LampCeiling size={26} strokeWidth={1.3} />
            <div className="flex-1">
              <h3 className="text-sm font-medium">Your connected space</h3>
              <p className="text-fg-muted mt-1 text-xs">
                Connect Home Assistant to control your home.
              </p>
            </div>
            <ArrowUpRight size={17} />
          </div>
        </Panel>
      </Link>
      <Link
        href="/memory"
        onClick={requestMemory}
        className={buttonVariants({ block: true, variant: 'outline' })}
      >
        <Plus size={17} />
        Save a memory
      </Link>
      <p className="text-fg-muted pb-2 text-center text-[10px] tracking-[.15em]">
        DESIGNED TO FEEL HUMAN
      </p>
    </Page>
  );
}
