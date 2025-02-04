import sounddevice as sd
import vosk
import json
import queue
import numpy as np

def main():
    # List available audio devices.
    print("Available audio devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device['name']}")
    
    # Find the Virtual Cable device.
    search_string = "Cable Output"  # Adjust if needed
    virtual_device_index = None
    for idx, device in enumerate(devices):
        if search_string.lower() in device['name'].lower():
            virtual_device_index = idx
            break

    if virtual_device_index is None:
        raise ValueError("Could not find the Cable Output device. "
                         "Please ensure VB-Audio Virtual Cable is installed and configured correctly.")

    print(f"\nUsing device #{virtual_device_index}: {devices[virtual_device_index]['name']}")
    device_info = sd.query_devices(virtual_device_index)
    print("\nDevice info:")
    print(device_info)

    # Recording parameters.
    samplerate = 16000     # Hz
    channels = 2            # Stereo (will be downmixed to mono)
    blocksize = 512         # Lower block size for lower latency

    # Load Vosk model.
    model_path = "vosk-model-small-ru-0.22"  # Update if using a larger model
    #model_path = "vosk-model-ru-0.42"
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    recognizer.SetWords(True)  # Enable word-level recognition

    # Create a queue for audio data.
    q = queue.Queue()

    # Amplification parameters (as before).
    desired_rms = 50.0
    max_gain = 100.0

    # Callback: downmix, amplify if needed, and enqueue audio.
    def audio_callback(indata, frames, time_info, status):
        if status:
            # Optionally print status.
            print("Audio status:", status, flush=True)
        # Convert raw bytes to an np.int16 array.
        audio_data = np.frombuffer(indata, dtype=np.int16)
        # Downmix stereo to mono by averaging channels.
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = ((audio_data[:, 0].astype(np.int32) + audio_data[:, 1].astype(np.int32)) // 2).astype(np.int16)
        # Compute RMS amplitude.
        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        if rms > 0 and rms < desired_rms:
            gain = min(desired_rms / rms, max_gain)
            audio_data = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
        q.put(audio_data.tobytes())

    # Variable to track previous partial result.
    prev_partial = ""

    try:
        with sd.RawInputStream(samplerate=samplerate,
                               blocksize=blocksize,
                               dtype="int16",
                               channels=channels,
                               callback=audio_callback,
                               device=virtual_device_index):
            print("\nRecording for continuous word-by-word recognition... Press Ctrl+C to stop.")
            while True:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Feed audio to the recognizer.
                if recognizer.AcceptWaveform(data):
                    # Final result produced—clear the partial state.
                    _ = json.loads(recognizer.Result())
                    prev_partial = ""
                # Always update from the partial result.
                partial_result = json.loads(recognizer.PartialResult())
                current_partial = partial_result.get("partial", "").strip()
                if current_partial != prev_partial:
                    prev_words = prev_partial.split() if prev_partial else []
                    current_words = current_partial.split()
                    if len(current_words) > len(prev_words):
                        new_words = current_words[len(prev_words):]
                        for word in new_words:
                            print("Word:", word, flush=True)
                    prev_partial = current_partial
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print("\nAn error occurred:", e)

if __name__ == '__main__':
    main()
