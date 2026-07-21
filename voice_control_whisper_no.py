import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import threading
import queue
import random
import time
import traceback
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
ASR_CHUNK_SECONDS = 3.0
ASR_MAX_BUFFER_SECONDS = 6.0
WHISPER_LANGUAGE = None  # set to "no" to force Norwegian
VAD_ACTIVITY_LEVEL = 0.015
VAD_MIN_ACTIVE_RATIO = 0.08
VAD_MIN_RMS = 0.02
ASR_TARGET_PEAK = 0.8
ASR_MAX_CHUNK_GAIN = 8.0

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

    # Guard against missing global tuning constants in older deployed copies.
    vad_activity_level = float(globals().get("VAD_ACTIVITY_LEVEL", 0.015))
    vad_min_active_ratio = float(globals().get("VAD_MIN_ACTIVE_RATIO", 0.08))
    vad_min_rms = float(globals().get("VAD_MIN_RMS", 0.02))

    buffer = np.zeros(0, dtype=np.float32)
    chunk_samples = int(MIC_SAMPLE_RATE * ASR_CHUNK_SECONDS)
    max_buffer_samples = int(MIC_SAMPLE_RATE * ASR_MAX_BUFFER_SECONDS)

    while True:
        try:
            audio = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        buffer = np.concatenate([buffer, audio])

        if len(buffer) > max_buffer_samples:
            buffer = buffer[-max_buffer_samples:]

        while len(buffer) >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            abs_chunk = np.abs(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            peak = float(np.max(abs_chunk))
            active_ratio = float(np.mean(abs_chunk >= vad_activity_level))

            if rms < vad_min_rms or active_ratio < vad_min_active_ratio:
                print(
                    "ASR: skipping low-voice chunk "
                    f"(rms={rms:.4f}, active={active_ratio:.2%}, peak={peak:.4f})"
                )
                continue

            if peak > 1e-6:
                chunk_gain = min(ASR_MAX_CHUNK_GAIN, ASR_TARGET_PEAK / peak)
                chunk = np.clip(chunk * chunk_gain, -1.0, 1.0)
            else:
                chunk_gain = 1.0

            print(
                f"ASR: transcribing {len(chunk) / MIC_SAMPLE_RATE:.1f}s chunk "
                f"(rms={rms:.4f}, active={active_ratio:.2%}, gain={chunk_gain:.2f})..."
            )
            started = time.time()

            try:
                transcribe_kwargs = {
                    "fp16": False,
                    "temperature": 0,
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.9,
                }
                if WHISPER_LANGUAGE:
                    transcribe_kwargs["language"] = WHISPER_LANGUAGE

                result = model.transcribe(chunk, **transcribe_kwargs)
                raw_text = result.get("text", "")
                text = raw_text.strip()
                elapsed = time.time() - started
                print(f"ASR: done in {elapsed:.2f}s, raw={raw_text!r}")

                if text:
                    print("TEXT:", text)
                    execute_command(text)
            except Exception as exc:
                print("ASR ERROR:", repr(exc))
                traceback.print_exc()

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