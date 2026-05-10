'use client';

import { useEffect, useRef } from 'react';

interface AudioCaptureProps {
  isRecording:  boolean;
  onAudioChunk: (base64: string) => void;
  onPermissionChange?: (status: 'requesting' | 'granted' | 'denied') => void;
}

const SAMPLE_RATE = 16000;
// createScriptProcessor requires a power-of-2 buffer size (256–16384).
// 4096 / 16000 = 256ms per chunk — close enough to the 250ms target.
const BUFFER_SIZE = 4096;

export function AudioCapture({ isRecording, onAudioChunk, onPermissionChange }: AudioCaptureProps) {
  const streamRef       = useRef<MediaStream | null>(null);
  const audioCtxRef     = useRef<AudioContext | null>(null);
  const processorRef    = useRef<ScriptProcessorNode | null>(null);
  const onChunkRef      = useRef(onAudioChunk);
  const onPermissionRef = useRef(onPermissionChange);
  const isRecordingRef  = useRef(isRecording);
  onChunkRef.current    = onAudioChunk;
  onPermissionRef.current = onPermissionChange;
  isRecordingRef.current  = isRecording;

  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      onPermissionRef.current?.('requesting');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
        audioCtxRef.current = ctx;

        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(BUFFER_SIZE, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (!isRecordingRef.current) return;

          const f32 = e.inputBuffer.getChannelData(0);
          const i16 = new Int16Array(f32.length);
          for (let i = 0; i < f32.length; i++) {
            const clamped = Math.max(-1, Math.min(1, f32[i]));
            i16[i] = clamped < 0 ? clamped * 32768 : clamped * 32767;
          }
          const bytes = new Uint8Array(i16.buffer);
          let binary = '';
          for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
          onChunkRef.current(btoa(binary));
        };

        source.connect(processor);
        processor.connect(ctx.destination);
        onPermissionRef.current?.('granted');
      } catch (err) {
        console.error('[AudioCapture] mic error:', err);
        onPermissionRef.current?.('denied');
      }
    };

    init();

    return () => {
      cancelled = true;
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stop() {
    processorRef.current?.disconnect();
    processorRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  return null; // Headless component
}
