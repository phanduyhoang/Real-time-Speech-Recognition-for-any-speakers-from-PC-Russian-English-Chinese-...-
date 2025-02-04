import sounddevice as sd
import numpy as np

# Callback function to process audio blocks
def audio_callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    # Calculate the RMS amplitude for demonstration purposes
    rms = np.sqrt(np.mean(indata**2))
    print(f"RMS Amplitude: {rms:.4f}")

def main():
    # List all available audio devices
    print("Available audio devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device['name']}")

    # Try to find the Virtual Cable device
    search_string = "Cable Output"  # Adjust this if your device name is different (e.g., "CABLE Output")
    virtual_device_index = None
    for idx, device in enumerate(devices):
        if search_string.lower() in device['name'].lower():
            virtual_device_index = idx
            break

    if virtual_device_index is None:
        raise ValueError("Could not find the Cable Output device. "
                         "Please ensure VB-Audio Virtual Cable is installed and configured correctly.")

    print(f"\nUsing device #{virtual_device_index}: {devices[virtual_device_index]['name']}")

    # Optional: Display device info to check supported parameters
    device_info = sd.query_devices(virtual_device_index)
    print("\nDevice info:")
    print(device_info)

    # Set the recording parameters
    samplerate = 44100  # in Hz
    channels = 2        # stereo
    blocksize = 1024    # frames per block

    # Open an input stream from the Virtual Cable's output device
    try:
        with sd.InputStream(device=virtual_device_index,
                            channels=channels,
                            samplerate=samplerate,
                            blocksize=blocksize,
                            callback=audio_callback):
            print("\nRecording from Virtual Cable... Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except Exception as e:
        print("\nAn error occurred:", e)

if __name__ == '__main__':
    main()
