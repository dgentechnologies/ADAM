// firebaseAdmin.js — Firebase Admin SDK initialisation (singleton)

import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { CONFIG } from './config.js';

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId:   CONFIG.FIREBASE.PROJECT_ID,
      clientEmail: CONFIG.FIREBASE.CLIENT_EMAIL,
      privateKey:  CONFIG.FIREBASE.PRIVATE_KEY,
    }),
  });
}

import { getFirestore as adminGetFirestore } from 'firebase-admin/firestore';
import { getAuth as adminGetAuth } from 'firebase-admin/auth';

function getAdminApp() {
  return getApps()[0];
}

export function getFirestore() {
  return adminGetFirestore(getAdminApp(), CONFIG.FIREBASE.DATABASE_ID);
}

export function getAuth() {
  return adminGetAuth(getAdminApp());
}
