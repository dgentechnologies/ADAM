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

// Re-export getFirestore so other modules import from here
export { getFirestore } from 'firebase-admin/firestore';
export { getAuth }      from 'firebase-admin/auth';
