'use client';

import { useState } from 'react';

interface FormData {
  name:     string;
  email:    string;
  company:  string;
  use_case: string;
}

export function WaitlistForm() {
  const [form,      setForm]      = useState<FormData>({ name: '', email: '', company: '', use_case: '' });
  const [submitted, setSubmitted] = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.email) { setError('Email is required.'); return; }
    setLoading(true); setError('');
    try {
      const res = await fetch('/api/waitlist', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(form),
      });
      const data = await res.json() as { error?: string };
      if (!res.ok) throw new Error(data.error ?? 'Something went wrong');
      setSubmitted(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div className="rounded-2xl bg-sky-500/10 border border-sky-500/30 p-8 text-center space-y-3">
        <div className="text-3xl">✓</div>
        <p className="font-bold text-xl">You're on the list.</p>
        <p className="text-gray-400 text-sm">DGEN will reach out when ADAM is ready to ship.</p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1.5">Name</label>
          <input
            name="name" value={form.name} onChange={handleChange}
            placeholder="Tirthankar Dasgupta"
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/15 focus:border-sky-500 outline-none text-sm text-white placeholder-gray-600 transition"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1.5">Email *</label>
          <input
            name="email" type="email" value={form.email} onChange={handleChange} required
            placeholder="you@example.com"
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/15 focus:border-sky-500 outline-none text-sm text-white placeholder-gray-600 transition"
          />
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1.5">Company / Organization</label>
        <input
          name="company" value={form.company} onChange={handleChange}
          placeholder="DGEN Technologies"
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/15 focus:border-sky-500 outline-none text-sm text-white placeholder-gray-600 transition"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1.5">How do you plan to use ADAM?</label>
        <textarea
          name="use_case" value={form.use_case} onChange={handleChange} rows={3}
          placeholder="Office assistant, personal AI companion, smart home hub…"
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/15 focus:border-sky-500 outline-none text-sm text-white placeholder-gray-600 transition resize-none"
        />
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <button
        type="submit" disabled={loading}
        className="w-full py-3 bg-sky-500 hover:bg-sky-400 disabled:opacity-60 text-white rounded-xl font-bold text-sm transition"
      >
        {loading ? 'Joining…' : 'Join Waitlist'}
      </button>
    </form>
  );
}
