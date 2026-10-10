import type {
  FirestoreCreditBalance,
  FirestoreDevice,
  FirestoreLaptopPairing,
  FirestoreMemoryFact,
  FirestoreMemoryPerson,
  FirestoreUser,
} from '@adam/types';
import {
  collection,
  doc,
  getDoc,
  getDocs,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  where,
} from 'firebase/firestore';

import { getFirebaseFirestore } from './config';

/**
 * users/{uid}
 */
export async function getUser(uid: string): Promise<FirestoreUser | null> {
  const db = getFirebaseFirestore();
  const snap = await getDoc(doc(db, 'users', uid));
  return snap.exists() ? (snap.data() as FirestoreUser) : null;
}

export const getUserDoc = getUser;

export async function updateUser(uid: string, data: Partial<FirestoreUser>): Promise<void> {
  const db = getFirebaseFirestore();
  const safe = Object.fromEntries(['displayName','email','photoUrl'].filter(k=>k in data).map(k=>[k,(data as any)[k]]));
  await updateDoc(doc(db, 'users', uid), {...safe,updatedAt:serverTimestamp()});
}

/**
 * devices/{deviceId}
 * deviceId = the ADAM's serial number or UUID (e.g. "DGEN-ADAM-0007")
 */
export async function getDevice(deviceId: string): Promise<FirestoreDevice | null> {
  const db = getFirebaseFirestore();
  const snap = await getDoc(doc(db, 'devices', deviceId));
  return snap.exists() ? (snap.data() as FirestoreDevice) : null;
}

export const getDeviceDoc = getDevice;

/**
 * Queries devices owned by a given user (ownerUid == uid).
 */
export async function getDevicesForUser(uid: string): Promise<FirestoreDevice[]> {
  const db = getFirebaseFirestore();
  const devicesRef = collection(db, 'devices');
  const q = query(devicesRef, where('ownerUid', '==', uid));
  const snap = await getDocs(q);
  return snap.docs.map((d) => d.data() as FirestoreDevice);
}

/**
 * Claims a device for a user:
 * 1. Creates/updates devices/{deviceId} with ownerUid, hardwareSerial, etc.
 * 2. Adds deviceId to users/{ownerUid}.linkedDeviceIds via arrayUnion.
 */
export async function claimDevice(
  _deviceId: string,
  _deviceData: Partial<FirestoreDevice> & { ownerUid: string },
): Promise<void> {
  // Ownership is never established by caller-supplied ownerUid/hardwareSerial.
  // Registered hardware proof is required; use claimRegisteredDevice below.
  throw new Error('Register this physical ADAM and confirm possession before claiming it.');
}

/**
 * devices/{deviceId}/memoryFacts/{factId}
 */
export async function getMemoryFacts(deviceId: string): Promise<(FirestoreMemoryFact & { id: string })[]> {
  const db = getFirebaseFirestore();
  const factsRef = collection(db, 'devices', deviceId, 'memoryFacts');
  const snap = await getDocs(factsRef);
  return snap.docs.map((d) => {
    const data = d.data();
    return {
      factId: data.factId || d.id,
      category: data.category || 'general',
      content: data.content || data.value || '',
      confidence: data.confidence ?? 1.0,
      learnedAt: data.learnedAt || data.savedAt,
      source: data.source || 'manual',
      id: d.id,
    } as FirestoreMemoryFact & { id: string };
  });
}

export async function addMemoryFact(
  deviceId: string,
  fact: Omit<FirestoreMemoryFact, 'learnedAt' | 'factId'> & { factId?: string; learnedAt?: any },
  factId?: string,
): Promise<string> {
  const db = getFirebaseFirestore();
  const factRef = factId
    ? doc(db, 'devices', deviceId, 'memoryFacts', factId)
    : doc(collection(db, 'devices', deviceId, 'memoryFacts'));

  await setDoc(factRef, {
    factId: factRef.id,
    schemaVersion:1,createdAt:serverTimestamp(),updatedAt:serverTimestamp(),deleted:false,deletedAt:null,origin:'mobile',
    category: fact.category || 'general',
    content: fact.content,
    confidence: fact.confidence ?? 1.0,
    source: fact.source || 'manual',
    learnedAt: serverTimestamp(),
  });

  return factRef.id;
}

/**
 * devices/{deviceId}/memoryPeople/{personId}
 */
export async function getMemoryPeople(deviceId: string): Promise<(FirestoreMemoryPerson & { id: string })[]> {
  const db = getFirebaseFirestore();
  const peopleRef = collection(db, 'devices', deviceId, 'memoryPeople');
  const snap = await getDocs(peopleRef);
  return snap.docs.map((d) => {
    const data = d.data();
    return {
      personId: data.personId || d.id,
      name: data.name || '',
      relationship: data.relationship || 'Contact',
      faceEncodingId: data.faceEncodingId || null,
      notes: data.notes || '',
      firstSeen: data.firstSeen || data.lastSeenAt,
      lastSeen: data.lastSeen || data.lastSeenAt,
      id: d.id,
    } as FirestoreMemoryPerson & { id: string };
  });
}

export async function addMemoryPerson(
  deviceId: string,
  person: Omit<FirestoreMemoryPerson, 'firstSeen' | 'lastSeen' | 'personId'> & { personId?: string },
  personId?: string,
): Promise<string> {
  const db = getFirebaseFirestore();
  const personRef = personId
    ? doc(db, 'devices', deviceId, 'memoryPeople', personId)
    : doc(collection(db, 'devices', deviceId, 'memoryPeople'));

  await setDoc(personRef, {
    personId: personRef.id,
    schemaVersion:1,createdAt:serverTimestamp(),updatedAt:serverTimestamp(),deleted:false,deletedAt:null,origin:'mobile',
    name: person.name,
    relationship: person.relationship || 'Contact',
    faceEncodingId: person.faceEncodingId || null,
    notes: person.notes || '',
    firstSeen: serverTimestamp(),
    lastSeen: serverTimestamp(),
  });

  return personRef.id;
}

/**
 * devices/{deviceId}/laptopPairings/{pairingId}
 */
export async function getLaptopPairings(deviceId: string): Promise<(FirestoreLaptopPairing & { id: string })[]> {
  const db = getFirebaseFirestore();
  const pairingsRef = collection(db, 'devices', deviceId, 'laptopPairings');
  const snap = await getDocs(pairingsRef);
  return snap.docs.map((d) => {
    const data = d.data();
    return {
      pairingId: data.pairingId || d.id,
      hostname: data.hostname || data.laptopName || 'Laptop',
      os: data.os || 'Unknown',
      tailscaleIp: data.tailscaleIp || '',
      pairedAt: data.pairedAt,
      lastActive: data.lastActive || data.lastActiveAt,
      id: d.id,
    } as FirestoreLaptopPairing & { id: string };
  });
}

export async function addLaptopPairing(
  deviceId: string,
  pairing: Omit<FirestoreLaptopPairing, 'pairedAt' | 'lastActive' | 'pairingId'> & { pairingId?: string },
  pairingId?: string,
): Promise<string> {
  const db = getFirebaseFirestore();
  const pairingRef = pairingId
    ? doc(db, 'devices', deviceId, 'laptopPairings', pairingId)
    : doc(collection(db, 'devices', deviceId, 'laptopPairings'));

  await setDoc(pairingRef, {
    pairingId:pairingRef.id,clientId:pairingRef.id,name:pairing.hostname,
    os:'windows',lastSeen:serverTimestamp(),updatedAt:serverTimestamp(),
  });

  return pairingRef.id;
}

/**
 * creditBalances/{deviceId}
 *
 * CRITICAL CONSTRAINT:
 * WRITE-PROTECTED on client. Client can only read.
 */
export async function getCreditBalance(deviceId: string): Promise<FirestoreCreditBalance | null> {
  const db = getFirebaseFirestore();
  const snap = await getDoc(doc(db, 'creditBalances', deviceId));
  if (!snap.exists()) return null;
  const data = snap.data();
  return {
    deviceId: data.deviceId || deviceId,
    balance: data.balance ?? data.balanceMinutes ?? 0,
    currency: 'credits',
    lastUpdated: data.lastUpdated || data.lastUpdatedAt,
    autoRecharge: !!data.autoRecharge,
    rechargeThreshold: data.rechargeThreshold ?? null,
  } as FirestoreCreditBalance;
}

/** Proof is produced by registered hardware during a physical possession flow. */
export async function claimRegisteredDevice(proof: {payload:string;signature:string}) {
  const {getFunctions, httpsCallable}=await import('firebase/functions');
  const {getFirebaseApp}=await import('./config');
  return (await httpsCallable(getFunctions(getFirebaseApp(),'us-central1'),'claimDevice')(proof)).data;
}
