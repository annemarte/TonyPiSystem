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
# NEW: auto-detect the ReSpeaker/Echo Pro input device by name instead of
# relying on a hardcoded index, since device indices can shift depending
# on what is plugged in / what pulse sees. Falls back to MIC_DEVICE_FALLBACK
# if no matching device name is found.
MIC_DEVICE_NAME_HINTS = ("echo", "respeaker", "seeed")
MIC_DEVICE_FALLBACK = 1


def _find_mic_device():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = dev.get("name", "").lower()
        if any(hint in name for hint in MIC_DEVICE_NAME_HINTS):
            return idx
    return MIC_DEVICE_FALLBACK


MIC_DEVICE = _find_mic_device()
mic_info = sd.query_devices(MIC_DEVICE, "input")
MIC_SAMPLE_RATE = int(mic_info["default_samplerate"])
print(f"Auto-selected mic device #{MIC_DEVICE}: {mic_info['name']!r} "
      f"({mic_info['max_input_channels']} input channels, "
      f"{MIC_SAMPLE_RATE} Hz)")
# NEW: Echo Pro / ReSpeaker devices often expose multiple mic channels
# (e.g. 4-6 channels for beamforming/raw mics). We open the stream with
# all of its input channels and mix them down to mono in the callback,
# instead of hardcoding channels=1 (which can fail to open or silently
# grab the wrong channel on multi-mic arrays).
MIC_CHANNELS = max(1, int(mic_info["max_input_channels"]))
MIC_GAIN = 1.0
ASR_CHUNK_SECONDS = 3.0
ASR_MAX_BUFFER_SECONDS = 6.0
WHISPER_LANGUAGE = None  # set to "no" to force Norwegian
VAD_ACTIVITY_LEVEL = 0.015
VAD_MIN_ACTIVE_RATIO = 0.08
VAD_MIN_RMS = 0.035
ASR_TARGET_PEAK = 0.8
ASR_MAX_CHUNK_GAIN = 8.0
ASR_NO_SPEECH_PROB_MAX = 0.6
ASR_MIN_ALPHA_CHARS = 3
ASR_HALLUCINATION_PHRASES = (
    "thank you", "thanks for watching", "subscribe", "bye", "bye bye",
    "you", "the", "okay", "so", "oh", "ah", "uh", "um", "hmm", "mm",
)

# NEW: background-noise handling.
NOISE_CALIBRATION_SECONDS = 2.0
NOISE_RMS_MARGIN = 1.5   # required speech rms multiple over measured noise floor
NOISE_ACTIVE_MARGIN = 1.2
NOISE_MAX_RMS_FLOOR = 0.15       # cap so a noisy calibration can't make VAD unreachable
NOISE_MAX_ACTIVE_RATIO_FLOOR = 0.5
# Absolute lower bounds. NOTE: these must be much smaller than VAD_MIN_RMS /
# VAD_MIN_ACTIVE_RATIO so that a genuinely quiet room can lower the
# calibrated thresholds below the static defaults. If the calibrated
# threshold were clamped to the static VAD_MIN_RMS/VAD_MIN_ACTIVE_RATIO as a
# lower bound, calibration would be a no-op whenever the room is quieter
# than those defaults (which is exactly the common case).
NOISE_MIN_RMS_FLOOR = 0.008
NOISE_MIN_ACTIVE_RATIO_FLOOR = 0.02
HIGHPASS_ALPHA = 0.98    # simple 1-pole high-pass to cut low-frequency rumble/hum

print("Using microphone:", mic_info["name"])
print("Microphone rate:", MIC_SAMPLE_RATE)

# CHANGED: bounded queue prevents unlimited latency and memory growth.
audio_queue = queue.Queue(maxsize=10)


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
    # NOTE: run aplay synchronously (no trailing "&") so we can wait for
    # playback to finish before the mic starts being used for ASR. This
    # prevents the ready sound itself (picked up by the mic through
    # speaker bleed) from being transcribed as a voice command.
    os.system(
        "aplay /home/pi/TonyPi/Functions/voice_interaction/audio/ready.wav "
        ">/dev/null 2>&1"
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

    # Drain any audio captured while the ready sound was playing (or
    # briefly queued right before/after it) so its own sound is never
    # fed into calibration or transcription.
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    # NEW: calibrate ambient background noise so VAD thresholds adapt
    # to the current room instead of using fixed values.
    vad_min_rms = VAD_MIN_RMS
    vad_min_active_ratio = VAD_MIN_ACTIVE_RATIO

    print(f"ASR: calibrating background noise for {NOISE_CALIBRATION_SECONDS:.1f}s...")
    noise_samples = []
    calib_deadline = time.time() + NOISE_CALIBRATION_SECONDS
    while time.time() < calib_deadline:
        try:
            noise_samples.append(audio_queue.get(timeout=0.5))
        except queue.Empty:
            continue

    total_calib_samples = sum(len(s) for s in noise_samples)
    print(
        f"ASR: calibration captured {len(noise_samples)} blocks "
        f"({total_calib_samples / MIC_SAMPLE_RATE:.2f}s of audio)"
    )

    if noise_samples:
        noise_audio = np.concatenate(noise_samples)
        noise_rms = float(np.sqrt(np.mean(noise_audio ** 2)))
        noise_active_ratio = float(np.mean(np.abs(noise_audio) >= VAD_ACTIVITY_LEVEL))

        # Clamp the calibrated thresholds so an unexpectedly loud/noisy
        # calibration window (e.g. echo/feedback, brief bump) can never
        # push the VAD thresholds above real-speech levels, which would
        # make speech unreachable (e.g. active_ratio > 100%). NOTE: the
        # lower bound is a small absolute floor (NOISE_MIN_RMS_FLOOR /
        # NOISE_MIN_ACTIVE_RATIO_FLOOR), NOT the static VAD_MIN_RMS /
        # VAD_MIN_ACTIVE_RATIO defaults -- otherwise calibration would be a
        # no-op in quiet rooms (the common case), which is what made it
        # look like calibration "wasn't working".
        vad_min_rms = min(
            NOISE_MAX_RMS_FLOOR,
            max(NOISE_MIN_RMS_FLOOR, noise_rms * NOISE_RMS_MARGIN),
        )
        vad_min_active_ratio = min(
            NOISE_MAX_ACTIVE_RATIO_FLOOR,
            max(NOISE_MIN_ACTIVE_RATIO_FLOOR, noise_active_ratio * NOISE_ACTIVE_MARGIN),
        )
        print(
            "ASR: noise floor "
            f"(rms={noise_rms:.4f}, active={noise_active_ratio:.2%}) -> "
            f"vad_min_rms={vad_min_rms:.4f}, vad_min_active_ratio={vad_min_active_ratio:.2%}"
        )
    else:
        print("ASR: no audio captured during noise calibration, using defaults")

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
            active_ratio = float(np.mean(abs_chunk >= VAD_ACTIVITY_LEVEL))

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
                    "no_speech_threshold": 0.6,
                }
                if WHISPER_LANGUAGE:
                    transcribe_kwargs["language"] = WHISPER_LANGUAGE

                result = model.transcribe(chunk, **transcribe_kwargs)
                raw_text = result.get("text", "")
                text = raw_text.strip()
                elapsed = time.time() - started

                segments = result.get("segments") or []
                if segments:
                    avg_no_speech_prob = float(
                        np.mean([seg.get("no_speech_prob", 0.0) for seg in segments])
                    )
                else:
                    avg_no_speech_prob = 0.0

                alpha_chars = sum(ch.isalpha() for ch in text)

                print(
                    f"ASR: done in {elapsed:.2f}s, raw={raw_text!r}, "
                    f"no_speech_prob={avg_no_speech_prob:.2f}, alpha_chars={alpha_chars}"
                )

                if not text:
                    continue

                if alpha_chars < ASR_MIN_ALPHA_CHARS:
                    print("ASR: ignoring non-speech/punctuation-only transcription")
                    continue

                normalized_text = text.strip(" .,!?").lower()
                if normalized_text in ASR_HALLUCINATION_PHRASES:
                    print("ASR: ignoring known hallucination phrase", repr(text))
                    continue

                if avg_no_speech_prob > ASR_NO_SPEECH_PROB_MAX and active_ratio < VAD_MIN_ACTIVE_RATIO * 2:
                    print("ASR: ignoring likely silence/hallucination (high no_speech_prob)")
                    continue

                print("TEXT:", text)
                execute_command(text)
            except Exception as exc:
                print("ASR ERROR:", repr(exc))
                traceback.print_exc()

# ======================
# Audio callback
# ======================
_hp_prev_in = 0.0
_hp_prev_out = 0.0


def audio_callback(indata, frames, time_info, status):
    global last_level_print_time, _hp_prev_in, _hp_prev_out

    if status:
        print("AUDIO STATUS:", status)

    audio = indata.copy()
    # NEW: mix multi-channel Echo Pro / ReSpeaker input down to mono by
    # averaging channels, instead of assuming a single-channel stream
    # (which may fail to open, or silently pick channel 0 only).
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio.flatten()

    # NEW: simple 1-pole high-pass filter to cut low-frequency background
    # rumble/hum (e.g. fans, AC) before amplification, carrying filter
    # state across callback blocks.
    filtered = np.empty_like(audio)
    prev_in = _hp_prev_in
    prev_out = _hp_prev_out
    for i, sample in enumerate(audio):
        out = HIGHPASS_ALPHA * (prev_out + sample - prev_in)
        filtered[i] = out
        prev_in = sample
        prev_out = out
    _hp_prev_in = prev_in
    _hp_prev_out = prev_out
    audio = filtered

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
        channels=MIC_CHANNELS,
        dtype="float32",
        blocksize=4000,
        callback=audio_callback,
):
    while True:
        time.sleep(0.1)