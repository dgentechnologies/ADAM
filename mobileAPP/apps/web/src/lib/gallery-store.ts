export interface Moment {
  id: string;
  title: string;
  createdAt: string;
  blob: Blob;
}
const DB = 'adam-moments-v1';
function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('moments', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(new Error('Photo storage could not be opened. Please restart the app.'));
    request.onblocked = () => reject(new Error('Close other ADAM tabs and try again.'));
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('moments', mode);
    const request = run(tx.objectStore('moments'));
    tx.oncomplete = () => {
      db.close();
      resolve(request.result);
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(tx.error ?? new Error('Could not save your photo. Check available storage.'));
    };
  });
}
export async function listMoments(): Promise<Moment[]> {
  const moments = await transaction('readonly', (store) => store.getAll());
  return (moments as Moment[]).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}
export async function normaliseImage(file: Blob, maxDimension = 1600): Promise<Blob> {
  if (!file.type.startsWith('image/')) throw new Error('Choose an image file.');
  if (file.size > 20 * 1024 * 1024) throw new Error('Choose a photo smaller than 20 MB.');
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Image processing is unavailable on this device.');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve, reject) =>
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('Could not process this image.'))),
        'image/jpeg',
        0.88,
      ),
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}
export async function addMoment(
  file: Blob,
  title: string,
  options: { id?: string } = {},
): Promise<void> {
  const blob = await normaliseImage(file);
  await transaction('readwrite', (store) =>
    store.put({
      id: options.id ?? crypto.randomUUID(),
      title: title.slice(0, 100),
      createdAt: new Date().toISOString(),
      blob,
    }),
  );
}
export async function deleteMoment(id: string) {
  await transaction('readwrite', (store) => store.delete(id));
}
export async function clearMoments() {
  await transaction('readwrite', (store) => store.clear());
}
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Could not read the photo.'));
    reader.readAsDataURL(blob);
  });
}
