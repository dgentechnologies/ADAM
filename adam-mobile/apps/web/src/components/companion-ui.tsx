'use client';

import { Button, Screen } from '@adam/ui';
import { AlertCircle, Bluetooth, ChevronRight, Loader2, type LucideIcon } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useId, useRef, type ReactNode } from 'react';
import { AppBar } from './app-bar';

export function Page({
  title,
  back,
  action,
  children,
}: {
  title: string;
  back?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <>
      <AppBar title={title} back={back} action={action} />
      <Screen chrome="both">
        <div className="page-stack">{children}</div>
      </Screen>
    </>
  );
}
export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>;
}
export function Notice({ children, error = false }: { children: ReactNode; error?: boolean }) {
  return (
    <div role={error ? 'alert' : 'status'} className={`notice ${error ? 'notice-error' : ''}`}>
      <AlertCircle size={17} aria-hidden />
      <div>{children}</div>
    </div>
  );
}
export function Loading() {
  return (
    <div role="status" className="text-fg-muted flex items-center justify-center gap-3 py-12">
      <Loader2 className="animate-spin" size={20} />
      Loading…
    </div>
  );
}
export function Empty({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: LucideIcon;
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="icon-orbit">
        <Icon size={28} strokeWidth={1.35} aria-hidden />
      </span>
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Row({
  href,
  icon: Icon,
  title,
  detail,
}: {
  href: string;
  icon: LucideIcon;
  title: string;
  detail?: string;
}) {
  return (
    <Link href={href} className="settings-row">
      <Icon size={21} strokeWidth={1.5} aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block break-words text-sm font-medium">{title}</span>
        {detail && (
          <span className="text-fg-muted mt-1 block break-words text-xs leading-relaxed">
            {detail}
          </span>
        )}
      </span>
      <ChevronRight size={17} className="text-fg-muted" aria-hidden />
    </Link>
  );
}
export function ConnectionNote() {
  return (
    <Link href="/device" className="connection-note">
      <Bluetooth size={18} strokeWidth={1.5} aria-hidden />
      <span>Explore your companion and its settings in the ADAM demo.</span>
    </Link>
  );
}
export function Dialog({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const id = useId();
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      dialog?.close();
      document.body.style.overflow = previous;
    };
  }, []);
  return (
    <dialog
      ref={ref}
      aria-labelledby={id}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      className="app-dialog"
    >
      <h2 id={id} className="text-lg font-semibold">
        {title}
      </h2>
      {children}
    </dialog>
  );
}
export function Confirm({
  title,
  children,
  onConfirm,
  onClose,
  busy = false,
}: {
  title: string;
  children: ReactNode;
  onConfirm: () => void;
  onClose: () => void;
  busy?: boolean;
}) {
  return (
    <Dialog
      title={title}
      onClose={() => {
        if (!busy) onClose();
      }}
    >
      <p className="text-fg-muted text-sm leading-relaxed">{children}</p>
      <Button block onClick={onConfirm} disabled={busy}>
        {busy ? 'Please wait…' : 'Confirm'}
      </Button>
      <Button block variant="ghost" onClick={onClose} disabled={busy}>
        Cancel
      </Button>
    </Dialog>
  );
}
