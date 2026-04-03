import speech_recognition as sr

# =========================
# 🔧 CONFIG
# =========================
PRODUCT_NAME = "adam"   # change this (kai, z, rei, etc.)
WAKE_PREFIXES = ["hey", "hello", "ok", "okay"]

# =========================
# 🔎 WAKE WORD CHECK
# =========================
def check_wake_word(text):
    words = text.lower().split()
    product = PRODUCT_NAME.lower()

    # strict match: "hey adam"
    for i in range(len(words) - 1):
        if words[i] in WAKE_PREFIXES and words[i+1] == product:
            return True

    # fallback: "adam" as standalone word
    if product in words:
        return True

    return False

# =========================
# 🎤 MAIN LOOP (FAST)
# =========================
def listen_loop():
    recognizer = sr.Recognizer()

    # ⚡ tuning for speed
    recognizer.pause_threshold = 0.6
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    print(f"🎤 Listening... Say 'Hey {PRODUCT_NAME}'")

    while True:
        with sr.Microphone() as source:
            try:
                # ⚡ fast capture
                audio = recognizer.listen(source, phrase_time_limit=2)

                text = recognizer.recognize_google(audio, language='en-IN')
                print("🗣️ You said:", text)

                if check_wake_word(text):
                    print(f"🔥 Wake word detected! ({PRODUCT_NAME})")

            except sr.UnknownValueError:
                # ignore noise
                pass
            except sr.RequestError as e:
                print("❌ API error:", e)

# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    listen_loop()