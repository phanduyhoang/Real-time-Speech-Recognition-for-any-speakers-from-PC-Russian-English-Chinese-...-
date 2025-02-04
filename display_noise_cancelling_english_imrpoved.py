import sounddevice as sd
import vosk
import json
import queue
import numpy as np
import tkinter as tk
from threading import Thread
import noisereduce as nr

class SpeechDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phan Duy Hoang Machine learning engineer")
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
        self.word_buffer = self.word_buffer[-self.max_words:]
        
        self.text_widget.delete(1.0, tk.END)
        
        if self.word_buffer:
            full_text = ' '.join(self.word_buffer[:-1]) + ' '
            self.text_widget.insert(tk.END, full_text, 'right')
            self.text_widget.insert(tk.END, self.word_buffer[-1], ('right', 'highlight'))
        
        self.text_widget.see(tk.END)
        self.root.update()

def main():
    devices = sd.query_devices()
    virtual_device_index = next(
        (idx for idx, dev in enumerate(devices) 
         if "cable output" in dev['name'].lower()),
        None
    )
    
    if virtual_device_index is None:
        raise ValueError("VB-Audio Virtual Cable not found!")

    # Audio processing parameters
    samplerate = 16000
    channels = 2
    blocksize = 2048

    # Adaptive noise reduction parameters
    NOISE_BUFFER_SIZE = 10  # Number of blocks to keep in noise profile (2 sec)
    RMS_THRESHOLD = 0.02    # Adjust based on environment (0.01-0.05)
    LEARNING_RATE = 0.1     # How quickly to adapt (0.0-1.0)
    current_gain = 1.0  
    MAX_GAIN = 5.0     
    TARGET_RMS = 0.1   
    GAIN_RAMP_RATE = 0.1  
    display = SpeechDisplay()
    model = vosk.Model("vosk-model-small-en-us-0.15")
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    q = queue.Queue()

    # Rolling noise profile buffer
    noise_buffer = []
    current_rms = 0.0

    def audio_callback(indata, frames, time_info, status):
        nonlocal noise_buffer, current_rms, current_gain
        
        # Convert to mono float32
        audio = np.frombuffer(indata, dtype=np.int16)
        if channels == 2:
            audio = audio.reshape(-1, 2)
            audio = np.mean(audio, axis=1)
        audio_float = audio.astype(np.float32) / 32768.0

        # Calculate current frame RMS
        frame_rms = np.sqrt(np.mean(audio_float**2))
        current_rms = (1-LEARNING_RATE)*current_rms + LEARNING_RATE*frame_rms

        # Update noise profile buffer
        if frame_rms < current_rms * 0.8:  # Only update during quiet moments
            noise_buffer.append(audio_float.copy())
            if len(noise_buffer) > NOISE_BUFFER_SIZE:
                noise_buffer.pop(0)

        # Apply noise reduction if we have a profile
        if len(noise_buffer) > 3:  # Minimum 3 frames (~0.3 sec)
            noise_profile = np.concatenate(noise_buffer)
            reduced_audio = nr.reduce_noise(
                y=audio_float,
                y_noise=noise_profile,
                sr=samplerate,
                stationary=False,
                prop_decrease=0.7
            )
        else:
            reduced_audio = audio_float  # Pass through while learning
        
        rms = np.sqrt(np.mean(reduced_audio**2)) + 1e-8
        
        # Dynamic gain control
        desired_gain = min(MAX_GAIN, TARGET_RMS / rms)
        current_gain = (1-GAIN_RAMP_RATE)*current_gain + GAIN_RAMP_RATE*desired_gain
        
        # Apply gain with safe clipping
        boosted_audio = np.clip(reduced_audio * current_gain, -1.0, 1.0)
        # Convert back to int16
        
        
        audio = (reduced_audio * 32767).astype(np.int16)
        q.put(audio.tobytes())

    def recognition_thread():
        prev_partial = ""
        try:
            with sd.RawInputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                dtype="int16",
                channels=channels,
                callback=audio_callback,
                device=virtual_device_index
            ):
                while True:
                    data = q.get(timeout=0.5)
                    if recognizer.AcceptWaveform(data):
                        _ = json.loads(recognizer.Result())
                    
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial != prev_partial:
                        current_words = partial.split()
                        new_words = current_words[len(prev_partial.split()):]
                        if new_words:
                            display.root.after(0, display.update_display, new_words)
                        prev_partial = partial

        except Exception as e:
            display.root.destroy()
            print(f"Error: {e}")

    Thread(target=recognition_thread, daemon=True).start()
    display.root.mainloop()

if __name__ == '__main__':
    main()