import type { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { AdamFacePreview } from '@/components/adam/AdamFacePreview';

export const metadata: Metadata = {
  title: 'ADAM — Autonomous Desktop AI Module',
  description:
    'ADAM by DGEN Technologies. Not a chatbot. Not a speaker. A presence. Powered by Gemini Live. Coming soon.',
};

const FEATURES = [
  {
    icon: '👁️',
    title: 'Sees You',
    description: 'Real-time camera vision and face recognition. ADAM knows who walked in — before you say a word.',
  },
  {
    icon: '🧠',
    title: 'Remembers You',
    description: 'Persistent memory across every session. Your preferences, your context, your history — always retained.',
  },
  {
    icon: '🇮🇳',
    title: 'Made in India',
    description: 'Designed and built by DGEN Technologies, Kolkata. Indigenous AI hardware for the world.',
  },
];

const TEAM = [
  { name: 'Tirthankar Dasgupta', role: 'CEO & CTO', slug: 'tirthankar-dasgupta' },
  { name: 'Sukomal Debnath',     role: 'CFO',       slug: 'sukomal-debnath' },
  { name: 'Sagnik Mandal',       role: 'CMO',       slug: 'sagnik-mandal' },
  { name: 'Arpan Bairagi',       role: 'COO',       slug: 'arpan-bairagi' },
];

export default function AdamLandingPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16 space-y-28">
      {/* Hero */}
      <section className="text-center space-y-6">
        <p className="text-sky-400 font-semibold text-sm uppercase tracking-widest">
          DGEN Technologies · Kolkata, India
        </p>
        <h1 className="text-6xl md:text-8xl font-black tracking-tight">ADAM</h1>
        <p className="text-xl md:text-2xl text-gray-300 font-medium">Autonomous Desktop AI Module</p>
        <p className="text-2xl md:text-3xl font-bold text-sky-400">
          Not a chatbot. Not a speaker.{' '}
          <span className="text-white">A presence.</span>
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          <Link
            href="/adam/demo"
            className="px-8 py-3 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-base font-bold transition shadow-lg shadow-sky-500/25"
          >
            Try ADAM (Beta)
          </Link>
          <Link
            href="/adam/waitlist"
            className="px-8 py-3 border-2 border-sky-500 hover:bg-sky-500/10 text-sky-400 rounded-xl text-base font-bold transition"
          >
            Join Waitlist
          </Link>
        </div>
      </section>

      {/* What is ADAM */}
      <section className="grid md:grid-cols-2 gap-10 items-center">
        <Image
          src="/images/adam-desktop-ai-module.png"
          alt="ADAM — Autonomous Desktop AI Module by DGEN Technologies"
          width={480}
          height={480}
          className="rounded-2xl mx-auto"
        />
        <div className="space-y-4">
          <h2 className="text-3xl font-bold">What is ADAM?</h2>
          <p className="text-gray-300 text-lg leading-relaxed">
            ADAM is a physical desk robot built by DGEN Technologies (founded 2025, Kolkata).
            Powered by Google's Gemini Live API, ADAM sees through its camera, speaks in real time,
            and remembers everything across sessions.
          </p>
          <p className="text-gray-400">
            It ships as hardware — servo neck, OLED animated face, persistent local memory — and as
            a browser demo you can try right now.
          </p>
        </div>
      </section>

      {/* Feature cards */}
      <section className="space-y-8">
        <h2 className="text-3xl font-bold text-center">Built different.</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="border border-sky-500/20 rounded-2xl p-6 bg-sky-950/10 hover:bg-sky-950/20 transition space-y-3"
            >
              <span className="text-4xl">{f.icon}</span>
              <h3 className="text-xl font-bold">{f.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Browser demo CTA */}
      <section className="text-center border border-sky-500/30 rounded-3xl p-12 bg-gradient-to-br from-sky-950/30 to-transparent space-y-6">
        <div className="flex justify-center">
          <AdamFacePreview />
        </div>
        <h2 className="text-3xl font-bold">Try it in your browser</h2>
        <p className="text-gray-400 max-w-lg mx-auto">
          No hardware required. Sign in with Google and talk to ADAM live — powered by Gemini Live,
          running on DGEN's servers. 5 minutes, 20 turns.
        </p>
        <Link
          href="/adam/demo"
          className="inline-block px-10 py-4 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-lg font-bold transition shadow-xl shadow-sky-500/20"
        >
          Launch Demo
        </Link>
      </section>

      {/* Team */}
      <section className="space-y-8">
        <h2 className="text-3xl font-bold text-center">The team behind ADAM</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {TEAM.map((m) => (
            <Link
              key={m.slug}
              href={`/about/${m.slug}`}
              className="border border-white/10 rounded-xl p-5 text-center hover:border-sky-500/50 hover:bg-sky-950/10 transition"
            >
              <div className="w-12 h-12 bg-sky-500/20 rounded-full mx-auto mb-3 flex items-center justify-center text-sky-400 font-bold text-lg">
                {m.name[0]}
              </div>
              <p className="font-semibold text-sm">{m.name}</p>
              <p className="text-gray-400 text-xs">{m.role}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* Waitlist CTA */}
      <section className="text-center space-y-6">
        <h2 className="text-3xl font-bold">ADAM ships soon.</h2>
        <p className="text-gray-400 max-w-md mx-auto">
          Be first in line. Join the waitlist and get early access when the hardware launches.
        </p>
        <Link
          href="/adam/waitlist"
          className="inline-block px-10 py-4 bg-white text-black rounded-xl text-lg font-bold hover:bg-gray-100 transition"
        >
          Join the Waitlist
        </Link>
      </section>
    </div>
  );
}
