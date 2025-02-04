import os
import json
import queue
import numpy as np
import tkinter as tk
import sounddevice as sd
from threading import Thread

import torch
from DTLN_pytorch.DTLN_model_copy import Pytorch_DTLN_stateful  # Ensure this module is available

import vosk

# ============ CONFIGURATION ============
RATE = 16000            # Sample rate (must match DTLN training and Vosk)
BLOCK_LEN = 512         # DTLN model input length (samples)
BLOCK_SHIFT = 128       # Processing step (samples)
RAW_BLOCKSIZE = 1024    # Samples per callback from sounddevice (must be a multiple of BLOCK_SHIFT)
CHANNELS = 2            # VB-Audio Virtual Cable typically outputs stereo

# When sending data to Vosk, accumulate at least this many bytes (fewer bytes => lower latency)
MIN_RECOGNIZER_BYTES = 2048  

# Path to your DTLN noise reduction model (update if necessary)
MODEL_PATH = "./DTLN_pytorch/pretrained/model.pth"

# ============ DEVICE SETUP ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ============ LOAD DTLN NOISE REDUCTION MODEL ============
print(f"🎯 Loading noise reduction model from: {MODEL_PATH}")
model = Pytorch_DTLN_stateful()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ============ SETUP TKINTER DISPLAY ============
class SpeechDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phan Duy Hoang Machine Learning Engineer")
        self.root.geometry("800x150")
        self.root.configure(bg='black')

        self.text_widget = tk.Text(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 24, "bold"),
            bg='black',
            fg='white',
            insertbackground='white',
            padx=20,
            pady=20
        )
        self.text_widget.pack(expand=True, fill=tk.BOTH)
        self.text_widget.tag_configure('right', justify='right')
        self.text_widget.tag_configure('highlight', foreground='#00FF00')
        self.text_widget.tag_configure('status', foreground='#808080')
        self.text_widget.insert(tk.END, "Play audio to see real-time transcription...", 'status')
        self.text_widget.configure(state='disabled')

        self.word_buffer = []
        self.max_words = 5
        self.first_update = True

    def update_display(self, new_words):
        if self.first_update:
            self.text_widget.configure(state='normal')
            self.text_widget.delete(1.0, tk.END)
            self.first_update = False

        self.word_buffer.extend(new_words)
        if len(self.word_buffer) > self.max_words:
            self.word_buffer = self.word_buffer[-self.max_words:]

        self.text_widget.delete(1.0, tk.END)
        if self.word_buffer:
            # Show all but the last word normally and the last word highlighted
            full_text = ' '.join(self.word_buffer[:-1]) + ' ' if len(self.word_buffer) > 1 else ''
            self.text_widget.insert(tk.END, full_text, 'right')
            self.text_widget.insert(tk.END, self.word_buffer[-1], ('right', 'highlight'))
        self.text_widget.see(tk.END)
        # Let the Tkinter mainloop handle the actual update (faster, with less overhead)
        self.root.update_idletasks()

# ============ SETUP VOSK SPEECH RECOGNIZER ============
# (Ensure that the model folder "vosk-model-small-ru-0.22" exists in your working directory.)
vosk_model = vosk.Model("vosk-model-small-ru-0.22")
recognizer = vosk.KaldiRecognizer(vosk_model, RATE)

# ============ AUDIO QUEUE ============
audio_queue = queue.Queue()

# ============ AUDIO CALLBACK ============
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Stream status:", status)
    # When using RawInputStream, indata is a bytes-like object.
    # Convert it to a NumPy array of int16.
    audio_data = np.frombuffer(indata, dtype=np.int16)
    try:
        # Reshape to (frames, CHANNELS)
        audio_data = audio_data.reshape(-1, CHANNELS)
    except ValueError as e:
        print("Reshape error:", e)
        return
    # Convert stereo to mono by averaging the channels
    mono_data = np.mean(audio_data, axis=1)
    # Now mono_data has RAW_BLOCKSIZE samples.
    # Split it into chunks of BLOCK_SHIFT samples and put them into the queue.
    num_chunks = len(mono_data) // BLOCK_SHIFT
    for i in range(num_chunks):
        start = i * BLOCK_SHIFT
        end = start + BLOCK_SHIFT
        chunk = mono_data[start:end]
        # Convert to float32 and normalize (int16 values -> [-1, 1])
        chunk = chunk.astype(np.float32) / 32768.0
        audio_queue.put(chunk)

# ============ PROCESSING THREAD (Noise Reduction + Recognition) ============
def processing_thread(display):
    # Buffers for overlap-add noise reduction
    in_buffer = np.zeros((BLOCK_LEN,), dtype=np.float32)
    out_buffer = np.zeros((BLOCK_LEN,), dtype=np.float32)

    # Initialize DTLN LSTM states (assumed shapes: (2, 1, 128))
    h1 = torch.zeros(2, 1, 128, dtype=torch.float32, device=device)
    c1 = torch.zeros(2, 1, 128, dtype=torch.float32, device=device)
    h2 = torch.zeros(2, 1, 128, dtype=torch.float32, device=device)
    c2 = torch.zeros(2, 1, 128, dtype=torch.float32, device=device)
    state1 = (h1, c1)
    state2 = (h2, c2)

    prev_partial = ""
    # Accumulate denoised audio bytes before sending to Vosk.
    recognizer_buffer = bytearray()

    while True:
        try:
            # Get a BLOCK_SHIFT-length chunk from the queue
            block = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Update the input buffer using an overlap-add strategy
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = block

        # Prepare input for the DTLN model (shape: [1, BLOCK_LEN])
        input_tensor = torch.from_numpy(in_buffer).unsqueeze(0).to(torch.float32).to(device)

        with torch.no_grad():
            output_tensor, state1, state2 = model(input_tensor, state1, state2)

        # Convert the model output to a NumPy array (shape: (BLOCK_LEN,))
        out_block_np = output_tensor.cpu().numpy().squeeze()

        # Overlap-add for synthesis: shift the output buffer and add the new output block
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = 0.0
        out_buffer += out_block_np

        # Extract the denoised output for this iteration (first BLOCK_SHIFT samples)
        denoised_block = out_buffer[:BLOCK_SHIFT]
        # Convert to int16 (scaled to 16-bit PCM) and then to bytes
        denoised_int16 = (denoised_block * 32768.0).astype(np.int16).tobytes()

        # Accumulate denoised audio for Vosk recognition
        recognizer_buffer.extend(denoised_int16)
        if len(recognizer_buffer) >= MIN_RECOGNIZER_BYTES:
            data = bytes(recognizer_buffer)
            if recognizer.AcceptWaveform(data):
                _ = json.loads(recognizer.Result())  # final result (if any) – here we ignore it
            partial_result = json.loads(recognizer.PartialResult()).get("partial", "").strip()
            if partial_result != prev_partial:
                current_words = partial_result.split()
                new_words = current_words[len(prev_partial.split()):]
                if new_words:
                    display.root.after(0, display.update_display, new_words)
                prev_partial = partial_result
            recognizer_buffer = bytearray()  # Reset the buffer after processing

# ============ MAIN FUNCTION ============
def main():
    # Find the VB-Audio Virtual Cable device by name (case-insensitive search for "cable output")
    devices = sd.query_devices()
    virtual_device_index = next(
        (idx for idx, dev in enumerate(devices) if "cable output" in dev['name'].lower()),
        None
    )
    if virtual_device_index is None:
        raise ValueError("VB-Audio Virtual Cable not found!")

    # Initialize the Tkinter display
    display = SpeechDisplay()

    # Start the processing thread (noise reduction + recognition)
    proc_thread = Thread(target=processing_thread, args=(display,), daemon=True)
    proc_thread.start()

    # Open the sounddevice RawInputStream using the virtual cable.
    try:
        stream = sd.RawInputStream(
            samplerate=RATE,
            blocksize=RAW_BLOCKSIZE,
            dtype='int16',
            channels=CHANNELS,
            device=virtual_device_index,
            callback=audio_callback
        )
        stream.start()
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return

    print("🎙️ Listening from VB-Audio Virtual Cable; speaking will trigger denoised, real-time transcription...")
    try:
        display.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

if __name__ == '__main__':
    main()
