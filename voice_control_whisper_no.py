import threading
import queue
import random
import os
import time
import numpy as np
import sounddevice as sd
import whisper

import hiwonder.ActionGroupControl as AGC
from hiwonder.Controller import Controller
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle

# ======================
# Robot init
# ======================
board = rrc.Board()
ctl = Controller(board)

servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
ctl.set_pwm_servo_pulse(1, 1500, 500)
ctl.set_pwm_servo_pulse(2, servo_data['servo2'], 500)
AGC.runActionGroup("stand")

# ======================
# Whisper init (Norwegian)
# ======================
print("Loading Whisper model...")
model = whisper.load_model("small")  # or "base"

SAMPLE_RATE = 16000
audio_queue = queue.Queue()

# ======================
# State
# ======================
dance_active = False
dance_thread = None
last_cmd_time = time.time()
CMD_COOLDOWN = 1.5

# ======================
# Audio feedback
# ======================
def play_ready_sound():
    os.system(
        "aplay /home/pi/TonyPi/Functions/voice_interaction/audio/ready.wav "
        ">/dev/null 2>&1 &"
    )

# ======================
# Dance loop (interruptible)
# ======================
def dance_loop():
    global dance_active
    while dance_active:
        AGC.runActionGroup(
            random.choice(["dance1", "dance2", "dance3", "dance4"]),
            1,
            False
        )
        time.sleep(0.05)

# ======================
# Command execution
# ======================
def execute_command(text):
    global dance_active, dance_thread, last_cmd_time

    text = text.lower().strip()
    print("CMD:", text)

    # --- STOP ALWAYS WINS ---
    if "stopp" in text or "stop" in text:
        print("EMERGENCY STOP")
        dance_active = False
        AGC.stopActionGroup()
        AGC.runActionGroup("stand")
        last_cmd_time = time.time()
        return

    # Cooldown for non-stop commands
    now = time.time()
    if now - last_cmd_time < CMD_COOLDOWN:
        return
    last_cmd_time = now

    if "dans" in text or "dance" in text or "God dag" in text:
        if not dance_active:
            print("Starting dance loop")
            dance_active = True
            dance_thread = threading.Thread(
                target=dance_loop,
                daemon=True
            )
            dance_thread.start()

    elif "frem" in text or "forward" in text:
        dance_active = False
        AGC.runActionGroup("go_forward", 2, True)

    elif "bak" in text or "tilbake" in text or "back" in text:
        dance_active = False
        AGC.runActionGroup("back", 2, True)

    elif "venstre" in text or "left" in text:
        dance_active = False
        AGC.runActionGroup("turn_left", 2, True)

    elif "høyre" in text or "right" in text:
        dance_active = False
        AGC.runActionGroup("turn_right", 2, True)

# ======================
# Whisper worker thread
# ======================
def whisper_worker():
    print("Whisper ASR ready (Norwegian)")
    play_ready_sound()

    buffer = np.zeros(0, dtype=np.float32)

    while True:
        audio = audio_queue.get()
        buffer = np.concatenate([buffer, audio])

        # Process ~2 seconds of audio
        if len(buffer) >= SAMPLE_RATE * 2:
            result = model.transcribe(
                buffer,
                language="no",
                fp16=False,
                temperature=0
            )

            text = result.get("text", "").strip()
            if text:
                print("TEXT:", text)
                execute_command(text)

            buffer = np.zeros(0, dtype=np.float32)

# ======================
# Audio callback
# ======================
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy().flatten())

# ======================
# Start threads
# ======================
threading.Thread(
    target=whisper_worker,
    daemon=True
).start()

print("Opening microphone...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback
):
    while True:
        time.sleep(0.1)
