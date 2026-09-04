'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'active'
  | 'denied'
  | 'unavailable'
  | 'error';

export interface UseFaceCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  status: CameraStatus;
  errorMessage: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  captureFrame: () => string | null;
}

/**
 * Custom hook that manages a front-facing camera stream via `getUserMedia`.
 *
 * Returns a `videoRef` to attach to a `<video>` element and a `canvasRef`
 * for capturing frames.  All cleanup is handled automatically on unmount.
 */
export function useFaceCamera(): UseFaceCameraReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<CameraStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus('idle');
  }, []);

  const startCamera = useCallback(async () => {
    if (status === 'active' || status === 'requesting') return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('unavailable');
      setErrorMessage('Your browser does not support camera access.');
      return;
    }

    setStatus('requesting');
    setErrorMessage(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 640 },
          aspectRatio: { ideal: 1 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (_) {
          // In some headless/embedded environments, play() resolves asynchronously
        }
      }

      setStatus('active');
    } catch (err) {
      const name = (err as Error)?.name ?? '';
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setStatus('denied');
        setErrorMessage(
          'Camera access was denied. Enable it in your browser settings and try again.',
        );
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setStatus('unavailable');
        setErrorMessage('No camera was found on this device.');
      } else {
        setStatus('error');
        setErrorMessage('Unable to start the camera. Please try again.');
      }
    }
  }, [status]);

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || status !== 'active') return null;

    const size = Math.min(video.videoWidth, video.videoHeight);
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const offsetX = (video.videoWidth - size) / 2;
    const offsetY = (video.videoHeight - size) / 2;
    ctx.drawImage(video, offsetX, offsetY, size, size, 0, 0, size, size);

    return canvas.toDataURL('image/png');
  }, [status]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, canvasRef, status, errorMessage, startCamera, stopCamera, captureFrame };
}
