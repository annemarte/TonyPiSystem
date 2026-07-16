#!/usr/bin/python3
# coding=utf-8

import json
import os
import queue
import random
import threading
import time

import sounddevice as sd
from vosk import KaldiRecognizer, Model

import hiwonder.ActionGroupControl as AGC
from hiwonder.Controller import Controller
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/home/pi/models/vosk-model-small-en-us-0.15"
READY_SOUND = (
    "/home/pi/TonyPi/Functions/voice_interaction/audio/ready.wav"
)

MICROPHONE_NAME = "USB PnP Audio Device"

DANCE_ACTIONS = [
    "dance1",
    "dance2",
    "dance3",
    "dance4",
]

COMMAND_COOLDOWN = 1.5
AUDIO_BLOCK_SIZE = 8000


# ============================================================
# Robot initialization
# ============================================================

board = rrc.Board()
ctl = Controller(board)

servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

ctl.set_pwm_servo_pulse(1, 1500, 500)
ctl.set_pwm_servo_pulse(2, servo_data["servo2"], 500)

AGC.runActionGroup("stand")


# ============================================================
# Microphone and Vosk initialization
# ============================================================
def find_microphone():
    default_input = sd.default.device[0]

    if default_input is not None and default_input >= 0:
        device = sd.query_devices(default_input, "input")
        return default_input, device

    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            return index, device

    raise RuntimeError("No input device found")


mic_device = 2
mic_info = sd.query_devices(mic_device, "input")
sample_rate = int(mic_info["default_samplerate"])

print(f"Microphone index: {mic_device}")
print(f"Microphone: {mic_info['name']}")
print(f"Sample rate: {sample_rate} Hz")

sd.check_input_settings(
    device=mic_device,
    samplerate=sample_rate,
    channels=1,
    dtype="int16",
)

print("Loading Vosk model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, sample_rate)


# ============================================================
# Shared state
# ============================================================

command_queue = queue.Queue()

dance_thread = None
dance_stop_event = threading.Event()
dance_lock = threading.Lock()

last_command_time = time.time() - COMMAND_COOLDOWN

audio_ready = False
audio_ready_lock = threading.Lock()


# ============================================================
# Utility functions
# ============================================================

def play_ready_sound():
    """Play the ready sound without blocking the main process."""
    if not os.path.isfile(READY_SOUND):
        print(f"Ready sound not found: {READY_SOUND}")
        return

    os.system(
        f'aplay "{READY_SOUND}" >/dev/null 2>&1 &'
    )


def stop_current_action(go_to_stand=True):
    global dance_thread

    dance_stop_event.set()
    AGC.stopActionGroup()

    if (
            dance_thread is not None
            and dance_thread.is_alive()
            and threading.current_thread() is not dance_thread
    ):
        dance_thread.join(timeout=2.0)

    if go_to_stand:
        AGC.runActionGroup("stand")
# ============================================================
# Dance handling
# ============================================================

def dance_loop():
    print("Dance loop started")

    try:
        if dance_stop_event.is_set():
            return

        action = random.choice(
            ["dance1", "dance2", "dance3", "dance4"]
        )

        print("DANCE ACTION:", action)

        with dance_lock:
            if dance_stop_event.is_set():
                return

            # Run one dance action only.
            AGC.runActionGroup(action, 1, False)

        # Poll while the action is running, then exit.
        for _ in range(100):
            if dance_stop_event.wait(0.05):
                break

    except Exception as error:
        print("Dance loop error:", error)

    finally:
        print("Dance loop stopped")

def start_dancing():
    global dance_thread

    if dance_thread is not None and dance_thread.is_alive():
        print("Already dancing")
        return

    dance_stop_event.clear()

    dance_thread = threading.Thread(
        target=dance_loop,
        name="dance-worker",
        daemon=True,
    )
    dance_thread.start()


# ============================================================
# Command handling
# ============================================================

def execute_command(text):
    global last_command_time

    text = text.lower().strip()

    if not text:
        return

    print("CMD:", text)

    # Stop always has the highest priority and bypasses cooldown.
    if "stop" in text or "stopp" in text or "no" in text or "nei" in text:
        print("Stopping immediately")
        stop_current_action(go_to_stand=True)
        last_command_time = time.time()
        return

    now = time.time()

    if now - last_command_time < COMMAND_COOLDOWN:
        print("Command ignored during cooldown")
        return

    last_command_time = now

    if "dance" in text or "dans" in text or "yo" in text:
        start_dancing()
        return

    # Any movement command first interrupts dancing.
    if any(
            word in text
            for word in (
                    "forward",
                    "back",
                    "backward",
                    "left",
                    "right",
                    "frem",
                    "bak",
                    "venstre",
                    "høyre",
            )
    ):
        stop_current_action(go_to_stand=False)

    if "forward" in text or "frem" in text:
        AGC.runActionGroup("go_forward", 2, True)

    elif (
            "backward" in text
            or "back" in text
            or "tilbake" in text
            or "bak" in text
    ):
        AGC.runActionGroup("back", 2, True)

    elif "left" in text or "venstre" in text:
        AGC.runActionGroup("turn_left", 2, True)

    elif "right" in text or "høyre" in text:
        AGC.runActionGroup("turn_right", 2, True)

    else:
        print(f"No command mapping for: {text}")


def command_worker():
    """Execute robot commands outside the audio callback."""
    while True:
        text = command_queue.get()

        try:
            execute_command(text)
        except Exception as error:
            print(f"Command execution error: {error}")
        finally:
            command_queue.task_done()


# ============================================================
# Audio processing
# ============================================================

def audio_callback(indata, frames, time_info, status):
    global audio_ready

    if status:
        print(f"Audio status: {status}")

    # Announce readiness only after real microphone data arrives.
    if not audio_ready:
        with audio_ready_lock:
            if not audio_ready:
                audio_ready = True
                print("Voice control ready. Speak commands...")
                play_ready_sound()

    try:
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()

            if text:
                print("TEXT:", text)
                command_queue.put(text)

    except Exception as error:
        print(f"Audio callback error: {error}")


# ============================================================
# Main
# ============================================================

def main():
    threading.Thread(
        target=command_worker,
        name="command-worker",
        daemon=True,
    ).start()

    print("Opening microphone...")

    try:
        with sd.RawInputStream(
                device=mic_device,
                samplerate=sample_rate,
                blocksize=AUDIO_BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=audio_callback,
        ):
            while True:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_current_action(go_to_stand=True)

    except Exception as error:
        print(f"Fatal error: {error}")
        stop_current_action(go_to_stand=True)
        raise


if __name__ == "__main__":
    main()