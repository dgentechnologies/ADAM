'use client';

import type { GalleryFilter, GalleryItem } from '@adam/types';
import { EmptyState, Screen, SegmentedControl, cn } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { ImageOff, Star } from 'lucide-react';
import { useMemo, useState } from 'react';

import { AppBar } from '@/components/app-bar';
import { fetchGallery, queryKeys } from '@/lib/mock/api';

/**
 * `moments` — the capture grid.
 *
 * No tile fetches an image: every Stitch source was a signed
 * `lh3.googleusercontent.com` URL that expires, and the mock fixtures carry an
 * empty `url` deliberately. Tiles therefore render the Digital Skin plus the
 * capture reason, which is also the honest empty-backend state.
 *
 * Starring is local-only in this pass — there is no mutation endpoint yet.
 */
const FILTERS: ReadonlyArray<{ value: GalleryFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'starred', label: 'Starred' },
  { value: 'this-week', label: 'This Week' },
];

const REASON_LABEL: Record<GalleryItem['reason'], string> = {
  requested: 'You asked',
  'face-recognised': 'Recognised',
  moment: 'Moment',
  scheduled: 'Scheduled',
};

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export default function GalleryPage() {
  const [filter, setFilter] = useState<GalleryFilter>('all');
  const [starred, setStarred] = useState<Record<string, boolean>>({});

  const { data: items = [], isPending } = useQuery({
    queryKey: queryKeys.gallery,
    queryFn: fetchGallery,
  });

  const isStarred = (item: GalleryItem) => starred[item.id] ?? item.starred;

  const visible = useMemo(() => {
    const now = Date.now();
    return items.filter((item) => {
      if (filter === 'starred') return isStarred(item);
      if (filter === 'this-week') return now - Date.parse(item.capturedAt) <= WEEK_MS;
      return true;
    });
    // `starred` is read through `isStarred`, so it belongs in the dep list; the
    // helper itself is re-created every render and must stay out of it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, filter, starred]);

  return (
    <>
      <AppBar title="Moments" />

      <Screen chrome="both">
        <div className="flex flex-col gap-stack-md">
          <SegmentedControl
            aria-label="Filter moments"
            options={FILTERS}
            value={filter}
            onChange={setFilter}
          />

          {isPending ? (
            <div className="grid grid-cols-2 gap-stack-sm">
              {Array.from({ length: 6 }, (_, index) => (
                <div
                  key={index}
                  className="aspect-square animate-breathe rounded-card border border-border bg-surface-raised"
                />
              ))}
            </div>
          ) : visible.length === 0 ? (
            <EmptyState
              icon={ImageOff}
              title="Nothing here yet."
              description={
                filter === 'starred'
                  ? 'Star a moment and it will show up here.'
                  : 'ADAM saves a frame when something worth remembering happens.'
              }
            />
          ) : (
            <div className="grid grid-cols-2 gap-stack-sm">
              {visible.map((item) => {
                const active = isStarred(item);
                return (
                  <figure
                    key={item.id}
                    className="relative aspect-square overflow-hidden rounded-card border border-border bg-surface-raised"
                  >
                    <div className="digital-skin-coarse absolute inset-0" aria-hidden />

                    <button
                      type="button"
                      aria-label={active ? 'Remove star' : 'Star this moment'}
                      aria-pressed={active}
                      onClick={() =>
                        setStarred((current) => ({ ...current, [item.id]: !active }))
                      }
                      className="absolute right-2 top-2 z-10 flex h-9 w-9 items-center justify-center rounded-full chrome-blur"
                    >
                      <Star
                        className={cn('h-4 w-4', active ? 'fill-fg text-fg' : 'text-fg-muted')}
                        strokeWidth={1.5}
                      />
                    </button>

                    <figcaption className="absolute inset-x-0 bottom-0 flex flex-col gap-0.5 p-stack-sm">
                      <span className="text-label-md text-fg">
                        {item.personName ?? REASON_LABEL[item.reason]}
                      </span>
                      <span className="text-label-xs uppercase text-fg-subtle">
                        {new Date(item.capturedAt).toLocaleDateString(undefined, {
                          day: 'numeric',
                          month: 'short',
                        })}
                      </span>
                    </figcaption>
                  </figure>
                );
              })}
            </div>
          )}
        </div>
      </Screen>
    </>
  );
}
