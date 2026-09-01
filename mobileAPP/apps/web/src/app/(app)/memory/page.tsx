'use client';

import type { MemoryKind } from '@adam/types';
import {
  Card,
  CardGroup,
  EmptyState,
  IconButton,
  ListRow,
  Screen,
  ScreenHeader,
  SegmentedControl,
} from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { BrainCircuit, ScanFace, Trash2, User } from 'lucide-react';
import { useState } from 'react';

import { AppBar } from '@/components/app-bar';
import { fetchMemory, queryKeys } from '@/lib/mock/api';

/**
 * `memory` — everything ADAM knows, with a delete on every row (spec §2.11).
 *
 * COLOUR: the Stitch markup styled this delete control `hover:text-error` /
 * `hover:bg-error-container/20`. That is the export's one real colour leak and it
 * is dropped rather than remapped — the theme has no `error` token, so it would
 * fail the Tailwind build, and destructive intent is carried by the icon plus the
 * confirm step instead of by hue.
 *
 * Deletion is local-only here; there is no mutation endpoint in this pass.
 */
const TABS: ReadonlyArray<{ value: MemoryKind; label: string }> = [
  { value: 'person', label: 'People' },
  { value: 'fact', label: 'Facts' },
];

export default function MemoryPage() {
  const [tab, setTab] = useState<MemoryKind>('person');
  const [deleted, setDeleted] = useState<string[]>([]);
  const [confirming, setConfirming] = useState<string | null>(null);

  const { data: entries = [], isPending } = useQuery({
    queryKey: queryKeys.memory,
    queryFn: fetchMemory,
  });

  const visible = entries.filter((entry) => entry.kind === tab && !deleted.includes(entry.id));

  return (
    <>
      <AppBar title="Memory" />

      <Screen chrome="both">
        <div className="flex flex-col gap-stack-md">
          <ScreenHeader
            size="xs"
            title={
              <>
                What ADAM
                <br />
                remembers
              </>
            }
            subtitle="People he has met and things you have told him. Forget anything, any time."
          />

          <SegmentedControl
            aria-label="Memory type"
            options={TABS}
            value={tab}
            onChange={(value) => {
              setTab(value);
              setConfirming(null);
            }}
          />

          {isPending ? (
            <CardGroup>
              {Array.from({ length: 3 }, (_, index) => (
                <div key={index} className="flex items-center gap-gutter px-stack-md py-gutter">
                  <span className="h-10 w-10 animate-breathe rounded-full bg-surface-pressed" />
                  <span className="h-4 w-40 animate-breathe rounded-full bg-surface-pressed" />
                </div>
              ))}
            </CardGroup>
          ) : visible.length === 0 ? (
            <EmptyState
              icon={tab === 'person' ? User : BrainCircuit}
              title={tab === 'person' ? 'No one yet.' : 'Nothing learned yet.'}
              description={
                tab === 'person'
                  ? 'ADAM adds a person once he has met them.'
                  : 'Tell ADAM something worth remembering and it will appear here.'
              }
            />
          ) : (
            <div className="flex flex-col gap-stack-sm">
              {visible.map((entry) => (
                <Card key={entry.id} padding="none">
                  <ListRow
                    className={entry.kind === 'fact' ? 'opacity-70' : undefined}
                    icon={
                      <span className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-surface">
                        {entry.kind === 'person' ? (
                          entry.hasFaceProfile ? (
                            <ScanFace
                              className="h-5 w-5"
                              strokeWidth={1.5}
                              aria-label="Face saved"
                            />
                          ) : (
                            <User className="h-5 w-5" strokeWidth={1.5} aria-hidden />
                          )
                        ) : (
                          <BrainCircuit className="h-5 w-5" strokeWidth={1.5} aria-hidden />
                        )}
                      </span>
                    }
                    title={entry.content}
                    trailing={
                      confirming === entry.id ? (
                        <span className="flex items-center gap-stack-sm">
                          <button
                            type="button"
                            onClick={() => {
                              setDeleted((current) => [...current, entry.id]);
                              setConfirming(null);
                            }}
                            className="rounded-full border border-fg bg-fg px-4 py-1.5 text-label-md text-fg-inverse"
                          >
                            Forget
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirming(null)}
                            className="text-label-md text-fg-muted"
                          >
                            Keep
                          </button>
                        </span>
                      ) : (
                        <IconButton
                          size="sm"
                          variant="ghost"
                          aria-label={`Forget ${entry.label}`}
                          onClick={() => setConfirming(entry.id)}
                        >
                          <Trash2 className="h-4 w-4" strokeWidth={1.5} />
                        </IconButton>
                      )
                    }
                  />
                </Card>
              ))}
            </div>
          )}

          <p className="text-label-md text-fg-faint">
            Memories live on the device. Forgetting one removes it from ADAM immediately.
          </p>
        </div>
      </Screen>
    </>
  );
}
