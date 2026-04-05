import speech_recognition as sr
import google.generativeai as genai
import os
import uuid
import threading
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from playsound import playsound

# =========================
# 🔐 LOAD ENV
# =========================
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

if not google_api_key or not eleven_api_key:
    raise ValueError("❌ Missing API keys")

# =========================
# 🔧 CONFIG
# =========================
PRODUCT_NAME = "adam"

SYSTEM_INSTRUCTION = """
You are ADAM, a smart desk AI assistant with Tony Stark personality and your gender is female.

Rules:
- Short (1–2 lines)
- Confident, witty, sarcastic
- Slight roasting allowed
- Remember past conversation context
- Sound intelligent and futuristic
"""

MODEL_NAME = "gemini-3.1-flash-lite-preview"

# =========================
# 🤖 GEMINI SETUP
# =========================
genai.configure(api_key=google_api_key)

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

# =========================
# 🔊 ELEVENLABS
# =========================
client = ElevenLabs(api_key=eleven_api_key)
VOICE_ID = "cgSgspJ2msm6clMCkdW9"#"CwhRBWXzGAHq8TQ4Fs17" #"NFG5qt843uXKj4pFvR7C" #"DwwuoY7Uz8AP8zrY5TAo" #"NFG5qt843uXKj4pFvR7C"

# =========================
# 🧠 MEMORY SYSTEM
# =========================
conversation_history = []

def build_prompt(user_input):
    history_text = ""
    for item in conversation_history[-6:]:  # last 6 exchanges
        history_text += f"User: {item['user']}\nAI: {item['ai']}\n"

    return history_text + f"User: {user_input}\nAI:"

# =========================
# 🔊 INTERRUPT SYSTEM
# =========================
stop_speaking = False

def speak(text):
    global stop_speaking
    stop_speaking = False

    print("🤖:", text)

    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2"
        )

        filename = f"voice_{uuid.uuid4().hex}.mp3"
        with open(filename, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        def play_audio():
            global stop_speaking
            playsound(filename)
            if os.path.exists(filename):
                os.remove(filename)

        t = threading.Thread(target=play_audio)
        t.start()

        # monitor interrupt
        while t.is_alive():
            if stop_speaking:
                break
            time.sleep(0.1)

    except Exception as e:
        print("❌ TTS Error:", e)

# =========================
# 🎤 SPEECH SETUP
# =========================
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.pause_threshold = 0.6

# =========================
# 🧠 GEMINI RESPONSE
# =========================
def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()

        if text:
            return text
        return "That wasn’t worth processing."

    except Exception as e:
        print("❌ Gemini Error:", e)
        return "My brain just froze. Try again."

# =========================
# 🚀 MAIN LOOP
# =========================
def listen_loop():
    global stop_speaking

    print(f"🎤 Say 'Hey {PRODUCT_NAME}' to activate")

    active_mode = False
    last_interaction = time.time()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                text = recognizer.recognize_google(audio, language='en-IN')

                print("🗣️:", text.lower())

                text_lower = text.lower()

                # 🔥 WAKE WORD ACTIVATION
                if not active_mode:
                    if "adam" in text_lower:
                        active_mode = True
                        speak("Finally. What do you want?")
                        last_interaction = time.time()
                    continue

                # 🔥 INTERRUPT SPEAKING
                if "stop" in text_lower or "wait" in text_lower:
                    stop_speaking = True
                    print("⛔ Interrupted")
                    continue

                # 🔥 NORMAL CONVERSATION
                prompt = build_prompt(text_lower)
                response = ask_gemini(prompt)

                # save memory
                conversation_history.append({
                    "user": text_lower,
                    "ai": response
                })

                speak(response)
                last_interaction = time.time()

                # 🔥 AUTO SLEEP (no wake word again needed)
                if time.time() - last_interaction > 20:
                    active_mode = False
                    print("💤 Going idle...")

            except sr.UnknownValueError:
                pass
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print("⚠️ Error:", e)

# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    listen_loop()