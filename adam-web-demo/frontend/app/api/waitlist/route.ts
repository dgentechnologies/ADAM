// app/api/waitlist/route.ts — saves a waitlist entry to Firestore

import { NextRequest } from 'next/server';
import { adminDb } from '@/lib/firebaseAdmin';
import { FieldValue } from 'firebase-admin/firestore';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json() as {
      email?: string;
      name?: string;
      company?: string;
      use_case?: string;
      referral?: string;
    };

    const { email, name, company, use_case, referral } = body;

    if (!email || typeof email !== 'string' || !email.includes('@')) {
      return Response.json({ error: 'Valid email is required' }, { status: 400 });
    }

    const normalised = email.toLowerCase().trim();
    const col        = adminDb.collection('waitlist');

    // Check for existing entry
    const existing = await col.where('email', '==', normalised).limit(1).get();
    if (!existing.empty) {
      // Idempotent — return success so the form doesn't show an error
      return Response.json({ success: true, alreadyRegistered: true });
    }

    await col.add({
      email:      normalised,
      name:       name      ?? '',
      company:    company   ?? '',
      useCase:    use_case  ?? '',
      referral:   referral  ?? '',
      signedUpAt: FieldValue.serverTimestamp(),
      confirmed:  false,
    });

    return Response.json({ success: true });
  } catch (err) {
    console.error('[waitlist] Unexpected error:', err);
    return Response.json({ error: 'Server error. Please try again.' }, { status: 500 });
  }
}
