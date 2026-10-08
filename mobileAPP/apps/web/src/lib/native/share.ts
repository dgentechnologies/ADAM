import { Capacitor } from '@capacitor/core';
import { blobToDataUrl } from '../gallery-store';

export async function clearShareCache(): Promise<void> {
  if (!Capacitor.isNativePlatform()) return;
  const { Filesystem, Directory } = await import('@capacitor/filesystem');
  const entries = await Filesystem.readdir({ path: '', directory: Directory.Cache });
  if (entries.files.some((entry) => entry.name === 'share'))
    await Filesystem.rmdir({ path: 'share', directory: Directory.Cache, recursive: true });
}

export async function shareFile(blob: Blob, name: string, title: string): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    const { Filesystem, Directory } = await import('@capacitor/filesystem');
    const { Share } = await import('@capacitor/share');
    // Android's chooser returns before the receiving app has read the URI.
    // Keep shared files for a day, then remove stale exports on the next share.
    const previous = await Filesystem.readdir({ path: 'share', directory: Directory.Cache }).catch(
      () => ({ files: [] }),
    );
    for (const file of previous.files) {
      const created = Number(file.name.split('-')[0]);
      if (created && Date.now() - created > 86400000)
        await Filesystem.deleteFile({
          path: `share/${file.name}`,
          directory: Directory.Cache,
        }).catch(() => undefined);
    }
    const path = `share/${Date.now()}-${name}`;
    const data = (await blobToDataUrl(blob)).split(',')[1] ?? '';
    const { uri } = await Filesystem.writeFile({
      path,
      data,
      directory: Directory.Cache,
      recursive: true,
    });
    await Share.share({ title, files: [uri], dialogTitle: title });
    return;
  }
  const file = new File([blob], name, { type: blob.type });
  if (navigator.canShare?.({ files: [file] })) {
    await navigator.share({ title, files: [file] });
    return;
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
