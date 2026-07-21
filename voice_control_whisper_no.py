import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import threading
import queue
import random
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
model = whisper.load_model("tiny")  # or "small", "base"

# Device 1 was "pulse".0 is echo pro. 1 is set to echo pro as source
MIC_DEVICE = 1
mic_info = sd.query_devices(MIC_DEVICE, "input")
MIC_SAMPLE_RATE = int(mic_info["default_samplerate"])
MIC_GAIN = 1.0

print("Using microphone:", mic_info["name"])
print("Microphone rate:", MIC_SAMPLE_RATE)

# CHANGED: bounded queue prevents unlimited latency and memory growth.
audio_queue = queue.Queue(maxsize=2)


# ======================
# State
# ======================
dance_active = False
dance_thread = None

CMD_COOLDOWN = 1.5
last_cmd_time = time.time() - CMD_COOLDOWN
last_level_print_time = 0.0

# ======================
# Audio feedback
# ======================
def play_ready_sound():
    os.system(
        "aplay /home/pi/TonyPi/Functions/voice_interaction/audio/ready.wav "
        ">/dev/null 2>&1 &"
    )

# ======================
# Dance action (single random dance)
# ======================
def run_random_dance_once():
    AGC.runActionGroup(
        random.choice(["dance1", "dance2", "dance3", "dance4"]),
        1,
        False
    )

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

    if "dans" in text or "dance" in text or "god dag" in text:
        print("Starting one random dance")
        dance_active = False
        run_random_dance_once()

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
        if len(buffer) >= MIC_SAMPLE_RATE * 2:
            print("ASR: transcribing audio chunk...")
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
    global last_level_print_time

    if status:
        print("AUDIO STATUS:", status)

    audio = indata.copy().flatten()

    # NEW: measure incoming signal level.
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # print the level at most twice per second.
    #now = time.time()
    #if now - last_level_print_time >= 0.5:
    #    print(f"MIC LEVEL: {rms:.5f}")
    #    last_level_print_time = now

    # NEW: software amplification for the weak microphone.
    audio = np.clip(audio * MIC_GAIN, -1.0, 1.0)

    # CHANGED: do not block the real-time audio callback.
    try:
        audio_queue.put_nowait(audio)

    except queue.Full:
        # Whisper is slower than incoming audio.
        # Drop the oldest block to avoid growing latency.
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except queue.Empty:
            pass

        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass


# ======================
# Start threads
# ======================
threading.Thread(
    target=whisper_worker,
    daemon=True
).start()

print("Available audio devices:")
print(sd.query_devices())

# NEW: verify that the selected device is a valid input.
mic_info = sd.query_devices(MIC_DEVICE, "input")
print("Using microphone:", mic_info["name"])

print("Opening microphone...")

with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=4000,
        callback=audio_callback,
):
    while True:
        time.sleep(0.1)