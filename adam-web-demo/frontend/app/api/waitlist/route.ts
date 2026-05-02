// app/api/waitlist/route.ts — saves a waitlist entry to Firestore

export const dynamic = 'force-dynamic';

import { NextRequest } from 'next/server';
import { adminDb } from '@/lib/firebaseAdmin';
import { FieldValue } from 'firebase-admin/firestore';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_FIELD_LENGTH = 300;

function normalizeField(value: unknown, maxLen = MAX_FIELD_LENGTH): string {
  if (typeof value !== 'string') return '';
  return value.trim().slice(0, maxLen);
}

export async function POST(req: NextRequest) {
  try {
    const contentType = req.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) {
      return Response.json({ error: 'Content-Type must be application/json' }, { status: 415 });
    }

    const body = await req.json() as {
      email?: string;
      name?: string;
      company?: string;
      use_case?: string;
      referral?: string;
    };

    const { email, name, company, use_case, referral } = body;

    const normalised = normalizeField(email, 320).toLowerCase();
    if (!normalised || !EMAIL_REGEX.test(normalised)) {
      return Response.json({ error: 'Valid email is required' }, { status: 400 });
    }

    const col        = adminDb.collection('waitlist');

    // Check for existing entry
    const existing = await col.where('email', '==', normalised).limit(1).get();
    if (!existing.empty) {
      // Idempotent — return success so the form doesn't show an error
      return Response.json({ success: true, alreadyRegistered: true });
    }

    await col.add({
      email:      normalised,
      name:       normalizeField(name),
      company:    normalizeField(company),
      useCase:    normalizeField(use_case, 800),
      referral:   normalizeField(referral),
      signedUpAt: FieldValue.serverTimestamp(),
      confirmed:  false,
    });

    return Response.json({ success: true });
  } catch (err) {
    console.error('[waitlist] Unexpected error:', err);
    return Response.json({ error: 'Server error. Please try again.' }, { status: 500 });
  }
}
