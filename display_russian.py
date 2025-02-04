import sounddevice as sd
import vosk
import json
import queue
import numpy as np
import tkinter as tk
from threading import Thread

class SpeechDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phan Duy Hoang Machine learning engineer")
        self.root.geometry("800x150")
        self.root.configure(bg='black')

        # Create text widget
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

        # Configure tags
        self.text_widget.tag_configure('right', justify='right')
        self.text_widget.tag_configure('highlight', foreground='#00FF00')
        self.text_widget.tag_configure('status', foreground='#808080')  # Gray color
        
        # Add initial status message
        self.text_widget.insert(tk.END, "Play audio to see real-time transcription...", 'status')
        self.text_widget.configure(state='disabled')
        
        self.word_buffer = []
        self.max_words = 5
        self.first_update = True  # Track first update

    def update_display(self, new_words):
        if self.first_update:
            # Clear initial message on first update
            self.text_widget.configure(state='normal')
            self.text_widget.delete(1.0, tk.END)
            self.first_update = False
        
        self.word_buffer.extend(new_words)
        if len(self.word_buffer) > self.max_words:
            self.word_buffer = self.word_buffer[-self.max_words:]
        
        self.text_widget.delete(1.0, tk.END)
        
        if self.word_buffer:
            full_text = ' '.join(self.word_buffer[:-1]) + ' '
            self.text_widget.insert(tk.END, full_text, 'right')
            self.text_widget.insert(tk.END, self.word_buffer[-1], ('right', 'highlight'))
        
        self.text_widget.see(tk.END)
        self.root.update()


def main():
    # Audio device setup
    devices = sd.query_devices()
    virtual_device_index = next(
        (idx for idx, dev in enumerate(devices) 
         if "cable output" in dev['name'].lower()),
        None
    )
    
    if virtual_device_index is None:
        raise ValueError("VB-Audio Virtual Cable not found!")

    # Audio configuration
    samplerate = 16000
    channels = 2
    blocksize = 2048

    # Initialize display
    display = SpeechDisplay()

    # Vosk setup
    model = vosk.Model("vosk-model-small-ru-0.22")
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        # Convert stereo to mono
        audio_data = np.frombuffer(indata, dtype=np.int16)
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = ((audio_data[:, 0] + audio_data[:, 1]) // 2).astype(np.int16)
        q.put(audio_data.tobytes())

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
                    
                    # Process audio through Vosk
                    if recognizer.AcceptWaveform(data):
                        _ = json.loads(recognizer.Result())  # Clear final results
                    
                    # Handle partial results
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial != prev_partial:
                        current_words = partial.split()
                        new_words = current_words[len(prev_partial.split()):]
                        
                        if new_words:
                            # Update display safely through main thread
                            display.root.after(0, display.update_display, new_words)
                        
                        prev_partial = partial

        except Exception as e:
            display.root.destroy()
            print(f"Error: {e}")

    # Start recognition thread
    Thread(target=recognition_thread, daemon=True).start()
    
    # Start GUI main loop
    display.root.mainloop()

if __name__ == '__main__':
    main()