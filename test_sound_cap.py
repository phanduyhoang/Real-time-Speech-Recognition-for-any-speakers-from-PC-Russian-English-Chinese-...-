import sounddevice as sd
import numpy as np

device_index = 4  # Try 31 (Stereo Mix) or any correct index from your list

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    print("Audio Level:", np.mean(indata))  # Shows volume levels

with sd.InputStream(device=device_index, channels=2, samplerate=44100, callback=callback):
    print(f"Listening on device {device_index}... (Press Ctrl+C to stop)")
    while True:
        pass
