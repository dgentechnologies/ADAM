'use client';

import { Button, Screen, ScreenHeader, TextField } from '@adam/ui';
import { ClipboardPaste, ExternalLink } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { sendByokKeyToDevice } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `connect_your_ai` — BYOK.
 *
 * SECURITY (tech spec §7): the key is handed to the unit over the local channel
 * encrypted with the unit's public key, and is never sent to `apps/api` nor
 * written to app storage. Nothing here persists it — not even in the Zustand
 * store, which is serialised to unencrypted preferences.
 *
 * The three steps sit on the page rather than inside a card (Stitch), with the
 * first numeral filled to read as "you are here"; the CTA sits under the field
 * instead of on the bottom rail so the whole instruction → paste → connect chain
 * stays in one visual block.
 */
const STEPS = [
  { n: 1, title: 'Tap below to create a free key' },
  { n: 2, title: 'Copy it' },
  { n: 3, title: 'Paste it here' },
] as const;

export default function ByokPage() {
  const router = useRouter();
  const complete = useSetupStore((state) => state.complete);
  const [key, setKey] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function connect() {
    setSending(true);
    setError(null);
    const { accepted } = await sendByokKeyToDevice(key);
    setSending(false);
    if (!accepted) {
      setError('ADAM rejected that key. Check you copied all of it.');
      return;
    }
    complete('byok');
    router.push('/camera-permission');
  }

  /** Best-effort: the Clipboard API is permission-gated and absent in some webviews. */
  async function paste() {
    try {
      const text = await navigator.clipboard.readText();
      if (text.trim()) setKey(text.trim());
    } catch {
      setError('Clipboard unavailable — paste with the keyboard instead.');
    }
  }

  return (
    <Screen className="pt-stack-md">
      <ScreenHeader
        size="xs"
        title="Connect your key"
        subtitle="Your key goes straight to ADAM, encrypted. It never touches our servers."
      />

      <div className="flex flex-col gap-stack-md pt-stack-lg">
        <ol className="flex flex-col gap-stack-sm">
          {STEPS.map(({ n, title }) => (
            <li key={n} className="flex items-center gap-gutter">
              <span
                className={
                  n === 1
                    ? 'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-fg font-display text-label-xs text-fg-inverse'
                    : 'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border-strong font-display text-label-xs text-fg-muted'
                }
              >
                {n}
              </span>
              <span className={n === 1 ? 'text-body-lg text-fg' : 'text-body-lg text-fg-muted'}>
                {title}
              </span>
            </li>
          ))}
        </ol>

        <Button
          block
          variant="outline"
          size="md"
          onClick={() =>
            window.open('https://aistudio.google.com/app/apikey', '_blank', 'noopener,noreferrer')
          }
        >
          Open Google AI Studio
          <ExternalLink className="h-4 w-4" strokeWidth={1.5} aria-hidden />
        </Button>

        <TextField
          placeholder="Paste your API key"
          aria-label="Gemini API key"
          autoComplete="off"
          spellCheck={false}
          value={key}
          onChange={(event) => setKey(event.target.value)}
          error={error ?? undefined}
          hint="Stored encrypted on ADAM. Never uploaded."
          trailing={
            <button
              type="button"
              onClick={() => void paste()}
              className="flex shrink-0 items-center gap-1.5 rounded-full border border-border-strong px-3 py-1.5 text-label-sm text-fg-muted transition-colors hover:text-fg"
            >
              <ClipboardPaste className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
              Paste
            </button>
          }
        />

        <Button
          block
          variant="primary"
          disabled={key.trim().length < 20 || sending}
          onClick={() => void connect()}
        >
          {sending ? 'Connecting ADAM…' : 'Connect ADAM'}
        </Button>
      </div>
    </Screen>
  );
}
