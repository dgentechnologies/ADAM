'use client';
import { Button, IconButton } from '@adam/ui';
import { Brain, Pencil, Plus, Search, Trash2, UserRound, ScanFace } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useMemoryIntent } from '@/stores/memory-intent';
import { Confirm, Dialog, Empty, Loading, Notice, Page, Panel } from '@/components/companion-ui';
import { deleteFact, errorMessage, saveFact, type Fact } from '@/lib/local-data';
import { useLocalData } from '@/lib/use-local-data';
import { stableDeviceId } from '@/lib/firebase/schema-documents';
export default function MemoryPage() {
  const { requested, clear } = useMemoryIntent();
  const { data, loading, error: loadError } = useLocalData();
  const [search, setSearch] = useState('');
  const [editing, setEditing] = useState<Fact | null | undefined>();
  const [removing, setRemoving] = useState<Fact | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (requested) {
      setEditing(null);
      clear();
    }
  }, [requested, clear]);
  const filtered = data.facts.filter((f) =>
    `${f.title} ${f.text}`.toLowerCase().includes(search.toLowerCase()),
  );
  async function remove() {
    if (!removing) return;
    setBusy(true);
    try {
      await deleteFact(removing.id);
      setRemoving(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page
      title="Memory"
      action={
        <IconButton
          aria-label="Add memory"
          variant="ghost"
          size="md"
          onClick={() => {
            setError('');
            setEditing(null);
          }}
        >
          <Plus size={22} />
        </IconButton>
      }
    >
      <div>
        <p className="eyebrow mb-3">WORTH REMEMBERING</p>
        <h2 className="page-title">
          Little things.
          <br />
          Lasting memories.
        </h2>
        <p className="text-fg-muted mt-3 text-sm leading-6">
          People, preferences, and thoughts. Saved on this phone, ready for life with ADAM.
        </p>
      </div>
      <label className="relative">
        <Search size={18} className="text-fg-muted absolute left-4 top-4" />
        <input
          className="field pl-11"
          aria-label="Search memories"
          placeholder="Search your memories"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </label>
      {(error || loadError) && <Notice error>{error || loadError}</Notice>}
      {loading ? (
        <Loading />
      ) : filtered.length === 0 ? (
        <Empty
          icon={Brain}
          title={search ? 'Nothing found' : 'A place for what matters'}
          action={
            !search && (
              <Button onClick={() => setEditing(null)}>
                <Plus size={17} />
                Add your first memory
              </Button>
            )
          }
        >
          {search
            ? 'Try a different word or name.'
            : 'Start with a favourite, an idea, or someone you want to remember.'}
        </Empty>
      ) : (
        <div className="space-y-3">
          {filtered.map((f) => (
            <Panel key={f.id}>
              <div className="mb-3 flex items-start gap-3">
                {f.kind === 'person' ? <UserRound size={19} /> : <Brain size={19} />}
                <h3 className="min-w-0 flex-1 break-words text-sm font-semibold">{f.title}</h3>
                <IconButton
                  size="md"
                  variant="ghost"
                  aria-label={`Edit ${f.title}`}
                  onClick={() => setEditing(f)}
                >
                  <Pencil size={16} />
                </IconButton>
                <IconButton
                  size="md"
                  variant="ghost"
                  aria-label={`Delete ${f.title}`}
                  onClick={() => {
                    setError('');
                    setRemoving(f);
                  }}
                >
                  <Trash2 size={16} />
                </IconButton>
              </div>
              <p className="text-fg-muted whitespace-pre-wrap break-words text-sm leading-6">
                {f.text}
              </p>
              <p className="text-fg-muted mt-4 text-[10px] uppercase tracking-wider">
                Saved {new Date(f.createdAt).toLocaleDateString()}
              </p>
            </Panel>
          ))}
        </div>
      )}
      <Link
        href="/face-capture"
        className="text-fg-muted flex min-h-12 items-center justify-center gap-2 text-sm"
      >
        <ScanFace size={17} />
        Your face profile
      </Link>
      {editing !== undefined && <MemoryForm fact={editing} onClose={() => setEditing(undefined)} />}
      {removing && (
        <Confirm
          title="Delete this memory?"
          onClose={() => setRemoving(null)}
          onConfirm={remove}
          busy={busy}
        >
          “{removing.title}” will be removed from this phone. This cannot be undone.
          {error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Page>
  );
}
function MemoryForm({ fact, onClose }: { fact: Fact | null; onClose: () => void }) {
  const { data } = useLocalData();
  const [deviceId, setDeviceId] = useState(fact?.deviceId ?? '');
  const [title, setTitle] = useState(fact?.title ?? '');
  const [text, setText] = useState(fact?.text ?? '');
  const [kind, setKind] = useState<Fact['kind']>(fact?.kind ?? 'fact');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim() || !text.trim() || busy) return;
    setBusy(true);
    try {
      await saveFact({ id: fact?.id, title, text, kind, deviceId });
      onClose();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog
      title={fact ? 'Edit memory' : 'A new memory'}
      onClose={() => {
        if (!busy) onClose();
      }}
    >
      <form onSubmit={submit} className="flex flex-col gap-4">
        <label className="field-label">Save for ADAM<select className="field" value={deviceId} disabled={Boolean(fact?.deviceId)} onChange={e=>setDeviceId(e.target.value)}><option value="">This phone only</option>{data.devices.map(device=><option key={device.id} value={stableDeviceId(device)}>{device.name}</option>)}</select></label>
        <p className="text-fg-muted text-xs">Choose an ADAM to sync this memory. Each ADAM keeps its own memories.</p>
        <label className="field-label">
          Title
          <input
            className="field"
            autoFocus
            required
            maxLength={80}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Your favourite coffee"
          />
        </label>
        <label className="field-label">
          Type
          <select
            className="field"
            value={kind}
            disabled={Boolean(fact?.deviceId)}
            onChange={(e) => setKind(e.target.value as Fact['kind'])}
          >
            <option value="fact">Thought or preference</option>
            <option value="person">Person</option>
          </select>
        </label>
        <label className="field-label">
          Details
          <textarea
            className="field min-h-28 resize-y"
            required
            maxLength={2000}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="The details you’d like to keep."
          />
        </label>
        {error && <Notice error>{error}</Notice>}
        <Button block type="submit" disabled={busy || !title.trim() || !text.trim()}>
          {busy ? 'Saving…' : 'Save memory'}
        </Button>
        <Button block variant="ghost" onClick={onClose} disabled={busy}>
          Cancel
        </Button>
      </form>
    </Dialog>
  );
}
