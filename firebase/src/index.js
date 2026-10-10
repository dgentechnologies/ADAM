import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { onCall, HttpsError } from 'firebase-functions/v2/https';
import { workflows } from './workflows.js';
import { Rejected } from './protocol.js';
initializeApp();
const service=workflows(getFirestore());
const callable = name => onCall({region:'us-central1',maxInstances:10,timeoutSeconds:30}, async request => {
  if (!request.auth) throw new HttpsError('unauthenticated','Sign in first.');
  try { return await service[name](request.auth.uid,request.data); }
  catch (e) {
    if (e instanceof Rejected) throw new HttpsError('failed-precondition',e.message);
    // Never log proof payloads, tokens, or private record contents.
    throw new HttpsError('internal','The verified operation could not be completed. Retry safely.');
  }
});
export const claimDevice=callable('claim');
export const transferDevice=callable('transfer');
export const reportExecution=callable('report');
