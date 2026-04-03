# 🔥 FINAL FIX: PROPER SPEECH SEGMENTATION (MIC START/STOP)
# Now behavior:
# ✅ Detect speech start
# ✅ Record until silence
# ✅ Send COMPLETE chunk to Gemini
# ✅ Then wait again

import asyncio
import os
import pyaudio
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# =====================
# ENV
# =====================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# =====================
# CONFIG
# =====================
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
RATE = 16000
CHUNK = int(RATE * 0.02)

client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription={},
    output_audio_transcription={},
)

# =====================
# AUDIO PLAYBACK
# =====================
def play_audio(pcm_bytes):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio, samplerate=24000)
    sd.wait()

# =====================
# MAIN
# =====================
async def run():
    print("🎤 Speak (auto start/stop detection)")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("✅ Connected")

        await session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part(text="You are ADAM, a smart assistant. Wait for user speech.")]
            )],
            turn_complete=True
        )

        async def sender():
            silence_threshold = 500
            silence_chunks = 20  # ~400ms silence end

            recording = False
            silence_count = 0
            audio_buffer = []

            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_np).mean()

                # 🎤 START speaking
                if volume > silence_threshold and not recording:
                    print("🎙️ Speech started")
                    recording = True
                    audio_buffer = []
                    silence_count = 0

                if recording:
                    audio_buffer.append(data)

                    if volume < silence_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    # 🛑 STOP speaking
                    if silence_count > silence_chunks:
                        print("🛑 Speech ended → sending to Gemini")
                        recording = False

                        # Send full chunk
                        for chunk in audio_buffer:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                            await asyncio.sleep(0.02)

                        audio_buffer = []

                await asyncio.sleep(0.02)

        async def receiver():
            buffer = b""

            async for response in session.receive():
                if not response.server_content:
                    continue

                sc = response.server_content

                if sc.input_transcription:
                    print("🗣️ YOU:", sc.input_transcription.text)

                if sc.output_transcription:
                    print("🤖 ADAM:", sc.output_transcription.text)

                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            buffer += part.inline_data.data

                if getattr(sc, "generation_complete", False):
                    if buffer:
                        play_audio(buffer)
                        buffer = b""

        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Stopped")