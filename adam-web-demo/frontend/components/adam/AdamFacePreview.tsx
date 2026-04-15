import { AdamFace } from './AdamFace';

export function AdamFacePreview({ size = 160 }: { size?: number }) {
  return <AdamFace emotion="idle" faceState="idle" size={size} />;
}
