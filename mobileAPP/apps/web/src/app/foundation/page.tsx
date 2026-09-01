import { palette, semantic } from '@adam/config/tokens';

/**
 * Foundation proof sheet — not one of the 22 product screens.
 *
 * It renders the token system so the theme can be reviewed before any screen is
 * built. Step 4.3 replaces this route with the splash redirect.
 */

const RAMP: Array<{ token: string; hex: string; note: string }> = [
  { token: 'black', hex: palette.black, note: 'app background' },
  { token: 'near-black', hex: palette['near-black'], note: 'recessed surface, inputs' },
  { token: 'charcoal', hex: palette.charcoal, note: 'elevated cards' },
  { token: 'charcoal-raised', hex: palette['charcoal-raised'], note: 'hairline on charcoal' },
  { token: 'grey-mid', hex: palette['grey-mid'], note: 'toggle track, strong hairline' },
  { token: 'grey', hex: palette.grey, note: 'placeholder, disabled' },
  { token: 'grey-light', hex: palette['grey-light'], note: 'secondary text' },
  { token: 'grey-lighter', hex: palette['grey-lighter'], note: 'tertiary on light' },
  { token: 'off-white', hex: palette['off-white'], note: 'light-mode surface' },
  { token: 'white', hex: palette.white, note: 'primary text and fills' },
];

const TYPE_SPECIMENS: Array<{ cls: string; label: string; sample: string }> = [
  { cls: 'text-display-lg', label: 'display-lg · 48/52 · -0.04em · 700', sample: 'Let’s wake him up.' },
  { cls: 'text-headline-md', label: 'headline-md · 36/40 · -0.03em · 700', sample: 'Choose ADAM’s brain.' },
  { cls: 'text-headline-sm', label: 'headline-sm · 32/36 · -0.02em · 700', sample: 'What ADAM remembers.' },
  { cls: 'text-title-md', label: 'title-md · 24/32 · -0.02em · 600', sample: 'Software Update' },
  { cls: 'text-body-lg', label: 'body-lg · 18/28 · -0.01em · 400', sample: 'Set up your ADAM in a few minutes.' },
  { cls: 'text-body-md', label: 'body-md · 16/24 · 0 · 400', sample: 'ADAM only supports 2.4GHz networks.' },
  { cls: 'text-label-md', label: 'label-md · 14/20 · +0.02em · 500', sample: 'Having trouble? Connect manually.' },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-stack-md">
      <h2 className="font-display text-label-sm uppercase text-fg-subtle">{title}</h2>
      {children}
    </section>
  );
}

export default function FoundationPage() {
  return (
    <main className="relative min-h-screen px-container pb-stack-lg pt-safe">
      <div className="pointer-events-none absolute inset-0 digital-skin-coarse" aria-hidden />

      <header className="relative flex flex-col gap-stack-sm pb-stack-lg pt-stack-lg">
        <p className="font-display text-label-xs uppercase text-fg-subtle">Step 4.1 · Foundation</p>
        <h1 className="text-display-lg text-fg">Achromatic system</h1>
        <p className="max-w-sm text-body-lg text-fg-muted">
          Ten-step ramp, one type scale, 8px grid. No hue token exists in the theme, so a stray
          coloured utility fails at build time.
        </p>
      </header>

      <div className="relative flex flex-col gap-stack-lg">
        <Section title="Ramp">
          <ul className="overflow-hidden rounded-card border border-border">
            {RAMP.map((step) => (
              <li
                key={step.token}
                className="flex items-center gap-gutter border-b border-border px-gutter py-stack-sm last:border-b-0"
              >
                <span
                  className="size-10 shrink-0 rounded-sm border border-border-strong"
                  style={{ backgroundColor: step.hex }}
                  aria-hidden
                />
                <span className="flex min-w-0 flex-col">
                  <span className="text-body-md text-fg">{step.token}</span>
                  <span className="text-label-md text-fg-muted">{step.note}</span>
                </span>
                <code className="ml-auto text-label-md uppercase text-fg-subtle">{step.hex}</code>
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Semantic aliases">
          <div className="grid grid-cols-2 gap-unit">
            {Object.entries(semantic).map(([name, hex]) => (
              <div key={name} className="rounded-md border border-border bg-surface-raised p-stack-sm">
                <p className="truncate text-label-md text-fg">{name}</p>
                <code className="text-label-md uppercase text-fg-subtle">{hex}</code>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Type scale">
          <div className="flex flex-col gap-stack-md">
            {TYPE_SPECIMENS.map((spec) => (
              <div key={spec.cls} className="flex flex-col gap-1">
                <p className="text-label-md text-fg-subtle">{spec.label}</p>
                <p className={`${spec.cls} text-fg`}>{spec.sample}</p>
              </div>
            ))}
            <div className="flex flex-col gap-1">
              <p className="text-label-md text-fg-subtle">
                font-display (Michroma) · wordmark + eyebrow only
              </p>
              <p className="font-display text-headline-md text-fg">ADAM</p>
              <p className="font-display text-label-sm uppercase text-fg-muted">Step 2 of 6</p>
            </div>
          </div>
        </Section>

        <Section title="Digital Skin">
          <div className="grid grid-cols-3 gap-unit">
            {[
              { cls: 'digital-skin', label: 'dot 16px' },
              { cls: 'digital-skin-coarse', label: 'dot 24px' },
              { cls: 'digital-skin-hatch', label: 'hatch 8px' },
            ].map((t) => (
              <div
                key={t.cls}
                className="relative aspect-square overflow-hidden rounded-card border border-border bg-surface-raised"
              >
                <div className={`absolute inset-0 ${t.cls}`} aria-hidden />
                <span className="absolute bottom-2 left-0 w-full text-center text-label-md text-fg-muted">
                  {t.label}
                </span>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Radius · elevation · hairline">
          <div className="flex flex-col gap-unit">
            <div className="rounded-card border border-border bg-surface-raised p-stack-md shadow-soft">
              <p className="text-body-md text-fg">card · 24px radius · shadow-soft</p>
              <p className="text-label-md text-fg-muted">0px 10px 30px rgba(0,0,0,0.5)</p>
            </div>
            <div className="rounded-sheet border border-border-strong bg-surface p-stack-md">
              <p className="text-body-md text-fg">sheet · 28px radius · strong hairline</p>
            </div>
            <div className="flex items-center gap-stack-sm">
              <button className="rounded-full bg-fg px-stack-md py-stack-sm text-label-md text-fg-inverse">
                Primary pill
              </button>
              <button className="rounded-full border border-fg px-stack-md py-stack-sm text-label-md text-fg">
                Secondary
              </button>
            </div>
          </div>
        </Section>

        <Section title="Status — shape carries meaning, never colour">
          <div className="flex items-center gap-stack-lg rounded-card border border-border bg-surface-raised p-stack-md">
            <span className="flex items-center gap-stack-sm">
              <span className="size-2.5 rounded-full bg-fg bloom" aria-hidden />
              <span className="text-label-md uppercase tracking-widest text-fg">Online</span>
            </span>
            <span className="flex items-center gap-stack-sm">
              <span className="size-2.5 rounded-full border border-fg-muted" aria-hidden />
              <span className="text-label-md uppercase tracking-widest text-fg-muted">Offline</span>
            </span>
            <span className="flex items-center gap-stack-sm">
              <span className="size-2.5 animate-breathe rounded-full bg-fg" aria-hidden />
              <span className="text-label-md uppercase tracking-widest text-fg-muted">Thinking</span>
            </span>
          </div>
        </Section>
      </div>
    </main>
  );
}
