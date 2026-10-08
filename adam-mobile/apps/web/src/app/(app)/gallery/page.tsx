'use client';
/* Local/offline images are sized before persistence; no image server is used. */
/* eslint-disable @next/next/no-img-element */
import { Button, IconButton } from '@adam/ui';
import { Capacitor } from '@capacitor/core';
import { Camera, Images, Plus, Share2, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Confirm, Dialog, Empty, Loading, Notice, Page } from '@/components/companion-ui';
import { addMoment, deleteMoment, listMoments, type Moment } from '@/lib/gallery-store';
import { errorMessage } from '@/lib/local-data';
import { shareFile } from '@/lib/native/share';
import { CameraNotice } from '@/components/camera-notice';
type Photo = Moment & { url: string };
export default function GalleryPage() {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<Photo | null>(null);
  const [confirm, setConfirm] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const captureInput = useRef<HTMLInputElement>(null);
  const urls = useRef<string[]>([]);
  const refresh = useCallback(async () => {
    try {
      const items = await listMoments();
      urls.current.forEach(URL.revokeObjectURL);
      const next = items.map((p) => ({ ...p, url: URL.createObjectURL(p.blob) }));
      urls.current = next.map((p) => p.url);
      setPhotos(next);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const onRecoveredPhoto = () => void refresh();
    window.addEventListener('adam:moments-changed', onRecoveredPhoto);
    return () => {
      window.removeEventListener('adam:moments-changed', onRecoveredPhoto);
      urls.current.forEach(URL.revokeObjectURL);
    };
  }, [refresh]);
  async function importFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setError('');
    try {
      for (const file of Array.from(files))
        await addMoment(file, file.name.replace(/\.[^.]+$/, ''));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      await refresh();
      setBusy(false);
      if (input.current) input.current.value = '';
      if (captureInput.current) captureInput.current.value = '';
    }
  }
  async function camera() {
    if (!Capacitor.isNativePlatform()) {
      captureInput.current?.click();
      return;
    }
    setBusy(true);
    setError('');
    try {
      const {
        Camera: NativeCamera,
        CameraResultType,
        CameraSource,
      } = await import('@capacitor/camera');
      const photo = await NativeCamera.getPhoto({
        quality: 90,
        resultType: CameraResultType.Uri,
        source: CameraSource.Camera,
        width: 1600,
        correctOrientation: true,
      });
      if (photo.webPath) {
        const blob = await (await fetch(photo.webPath)).blob();
        await addMoment(blob, 'A new moment');
        await refresh();
      }
    } catch (e) {
      if (!/cancel/i.test(errorMessage(e))) setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!selected) return;
    setBusy(true);
    try {
      await deleteMoment(selected.id);
      setConfirm(false);
      setSelected(null);
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function share() {
    if (!selected) return;
    setBusy(true);
    try {
      await shareFile(selected.blob, `adam-${selected.id}.jpg`, selected.title);
    } catch (e) {
      if (!/cancel|abort/i.test(errorMessage(e))) setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page
      title="Moments"
      action={
        <IconButton
          aria-label="Import photos"
          variant="ghost"
          size="md"
          disabled={busy}
          onClick={() => input.current?.click()}
        >
          <Plus size={22} />
        </IconButton>
      }
    >
      <div>
        <p className="eyebrow mb-3">LIFE, IN LITTLE FRAMES</p>
        <h2 className="page-title">
          Keep a little
          <br />
          of the everyday.
        </h2>
        <p className="text-fg-muted mt-3 text-sm leading-6">
          Your photos, privately saved on this phone.
        </p>
      </div>
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        aria-label="Choose photos"
        onChange={(e) => void importFiles(e.target.files)}
      />
      <input
        ref={captureInput}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        aria-label="Take photo"
        onChange={(e) => void importFiles(e.target.files)}
      />
      <div className="grid grid-cols-2 gap-3">
        <Button variant="outline" size="md" disabled={busy} onClick={camera}>
          <Camera size={17} />
          Take photo
        </Button>
        <Button variant="outline" size="md" disabled={busy} onClick={() => input.current?.click()}>
          <Plus size={17} />
          Import
        </Button>
      </div>
      {error && <CameraNotice error={error} />}
      {busy && (
        <p role="status" className="text-fg-muted text-sm">
          Working on your photo…
        </p>
      )}
      {loading ? (
        <Loading />
      ) : photos.length === 0 ? (
        <Empty icon={Images} title="Every moment starts somewhere">
          Take a photo or bring in a favourite from your phone. Keep the everyday close.
        </Empty>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          {photos.map((p) => (
            <button
              key={p.id}
              aria-label={`Open ${p.title}`}
              onClick={() => {
                setError('');
                setSelected(p);
              }}
              className="border-border bg-surface overflow-hidden rounded-2xl border text-left"
            >
              <img
                src={p.url}
                alt={p.title}
                className="aspect-square w-full object-cover"
                loading="lazy"
              />
              <div className="p-3">
                <p className="truncate text-xs font-medium">{p.title}</p>
                <p className="text-fg-muted mt-1 text-[10px]">
                  {new Date(p.createdAt).toLocaleDateString()}
                </p>
              </div>
            </button>
          ))}
        </div>
      )}
      {selected && !confirm && (
        <Dialog
          title={selected.title}
          onClose={() => {
            if (!busy) setSelected(null);
          }}
        >
          <img
            src={selected.url}
            alt={selected.title}
            className="max-h-[50dvh] w-full rounded-xl object-contain"
          />
          {error && <Notice error>{error}</Notice>}
          <Button block variant="outline" disabled={busy} onClick={share}>
            <Share2 size={17} />
            Share photo
          </Button>
          <Button block variant="ghost" disabled={busy} onClick={() => setConfirm(true)}>
            <Trash2 size={17} />
            Delete from ADAM
          </Button>
          <Button block variant="ghost" disabled={busy} onClick={() => setSelected(null)}>
            Close
          </Button>
        </Dialog>
      )}
      {selected && confirm && (
        <Confirm
          title="Delete this photo?"
          busy={busy}
          onClose={() => setConfirm(false)}
          onConfirm={remove}
        >
          This removes the copy saved in ADAM. The original in your phone’s gallery is unchanged.
          {error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Page>
  );
}
