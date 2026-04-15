'use client';

import { useEffect, useRef } from 'react';

interface AudioCaptureProps {
  isRecording:  boolean;
  onAudioChunk: (base64: string) => void;
}

const SAMPLE_RATE   = 16000;
const CHUNK_SIZE_MS = 250;

export function AudioCapture({ isRecording, onAudioChunk }: AudioCaptureProps) {
  const streamRef       = useRef<MediaStream | null>(null);
  const audioCtxRef     = useRef<AudioContext | null>(null);
  const processorRef    = useRef<ScriptProcessorNode | null>(null);
  const onChunkRef      = useRef(onAudioChunk);
  onChunkRef.current    = onAudioChunk;

  useEffect(() => {
    if (!isRecording) {
      stop();
      return;
    }
    start();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording]);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true } });
      streamRef.current   = stream;

      const ctx       = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ctx;

      const source    = ctx.createMediaStreamSource(stream);
      const bufSize   = Math.floor(SAMPLE_RATE * CHUNK_SIZE_MS / 1000);
      const processor = ctx.createScriptProcessor(bufSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const f32  = e.inputBuffer.getChannelData(0);
        const i16  = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
          const clamped = Math.max(-1, Math.min(1, f32[i]));
          i16[i] = clamped < 0 ? clamped * 32768 : clamped * 32767;
        }
        const bytes  = new Uint8Array(i16.buffer);
        let binary   = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        onChunkRef.current(btoa(binary));
      };

      source.connect(processor);
      processor.connect(ctx.destination);
    } catch (err) {
      console.error('[AudioCapture] mic error:', err);
    }
  }

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
