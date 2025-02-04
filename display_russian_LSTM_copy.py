import sounddevice as sd
import vosk
import json
import queue
import numpy as np
import tkinter as tk
from threading import Thread
import torch
from DTLN_pytorch.DTLN_model_copy import Pytorch_DTLN_stateful

# Configuration
MODEL_PATH = "./DTLN_pytorch/pretrained/model.pth"
BLOCK_LEN = 512
BLOCK_SHIFT = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phan Duy Hoang - Noise Reduced Transcription")
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
        
        self.word_buffer = []
        self.max_words = 5
        self.first_update = True
        self.text_widget.insert(tk.END, "Initializing...", 'status')
        self.text_widget.configure(state='disabled')

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

    # Create queues
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()

    # Vosk setup
    model = vosk.Model("vosk-model-small-ru-0.22")
    recognizer = vosk.KaldiRecognizer(model, samplerate)

    # DTLN model setup
    dtln_model = Pytorch_DTLN_stateful()
    dtln_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    dtln_model.to(DEVICE)
    dtln_model.eval()

    # Audio processing buffers and states
    in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    in_state1 = (
        torch.zeros(2, 1, 128, dtype=torch.float32, device=DEVICE),
        torch.zeros(2, 1, 128, dtype=torch.float32, device=DEVICE)
    )
    in_state2 = (
        torch.zeros(2, 1, 128, dtype=torch.float32, device=DEVICE),
        torch.zeros(2, 1, 128, dtype=torch.float32, device=DEVICE)
    )

    def audio_callback(indata, frames, time_info, status):
        audio_data = np.frombuffer(indata, dtype=np.int16)
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = ((audio_data[:, 0] + audio_data[:, 1]) // 2).astype(np.int16)
        raw_queue.put(audio_data.tobytes())

    def denoising_thread():
        nonlocal in_state1, in_state2, in_buffer, out_buffer
        while True:
            raw_data = raw_queue.get()
            raw_audio = np.frombuffer(raw_data, dtype=np.int16)
            audio_float = raw_audio.astype(np.float32) / 32768.0
            
            num_chunks = len(audio_float) // BLOCK_SHIFT
            denoised_audio = np.zeros(num_chunks * BLOCK_SHIFT, dtype=np.float32)
            
            for i in range(num_chunks):
                chunk = audio_float[i*BLOCK_SHIFT:(i+1)*BLOCK_SHIFT]
                
                in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
                in_buffer[-BLOCK_SHIFT:] = chunk
                
                input_tensor = torch.from_numpy(in_buffer).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    h1, c1 = in_state1
                    h2, c2 = in_state2
                    out_block, (h1, c1), (h2, c2) = dtln_model(input_tensor, (h1, c1), (h2, c2))
                    in_state1 = (h1.contiguous(), c1.contiguous())
                    in_state2 = (h2.contiguous(), c2.contiguous())
                
                out_block = out_block.cpu().numpy().squeeze()
                out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
                out_buffer[-BLOCK_SHIFT:] = np.zeros(BLOCK_SHIFT)
                out_buffer += out_block
                denoised_audio[i*BLOCK_SHIFT:(i+1)*BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]
            
            denoised_bytes = (denoised_audio * 32768.0).astype(np.int16).tobytes()
            processed_queue.put(denoised_bytes)

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
                Thread(target=denoising_thread, daemon=True).start()
                while True:
                    data = processed_queue.get(timeout=0.5)
                    
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