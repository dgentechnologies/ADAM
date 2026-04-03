import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# =========================
# 🔧 CONFIGURATION
# =========================
PRODUCT_NAME = "J"   # change this to test (KAI, Z, etc.)
WAKE_PREFIXES = ["hey", "hello", "ok", "okay"]

MODEL_PATH = "vosk-model-small-en-in-0.4"  # path to your model

# =========================
# 🎤 SETUP
# =========================
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# =========================
# 🔎 WAKE WORD CHECK
# =========================
def check_wake_word(text):
    text = text.lower()
    product = PRODUCT_NAME.lower()

    for prefix in WAKE_PREFIXES:
        if f"{prefix} {product}" in text:
            return True

    return False

# =========================
# 🚀 MAIN LOOP
# =========================
print(f"🎤 Listening... Wake word: 'Hey {PRODUCT_NAME}'")

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=audio_callback):

    while True:
        data = q.get()

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            if text:
                print("🗣️ You said:", text)

                if check_wake_word(text):
                    print(f"🔥 Wake word detected! ({PRODUCT_NAME})")