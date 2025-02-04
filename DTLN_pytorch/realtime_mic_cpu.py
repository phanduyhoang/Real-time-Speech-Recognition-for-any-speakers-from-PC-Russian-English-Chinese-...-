import os
import torch
import numpy as np
import pyaudio
from DTLN_model import Pytorch_DTLN_stateful

# === CONFIGURATION ===
MODEL_PATH = "./pretrained/model.pth"  # Update this path if necessary
RATE = 16000   # Sample rate (matches DTLN's training)
BLOCK_LEN = 512  # Number of samples per frame (32ms per frame)
BLOCK_SHIFT = 128  # Frame shift (8ms per step)
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio

# === LOAD MODEL ===
print(f"🔥 Loading model from: {MODEL_PATH}")
model = Pytorch_DTLN_stateful()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === INIT AUDIO STREAMS ===
p = pyaudio.PyAudio()

# Microphone input stream
stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                   frames_per_buffer=BLOCK_LEN)

# Speaker output stream
stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                    frames_per_buffer=BLOCK_LEN)

print("🎤 Real-time noise suppression running... Speak into the mic!")

# === REAL-TIME PROCESSING ===
# Buffers
in_buffer = np.zeros((BLOCK_LEN))
out_buffer = np.zeros((BLOCK_LEN))
in_state1 = torch.zeros(2, 1, 128, 2)  # LSTM state
in_state2 = torch.zeros(2, 1, 128, 2)

try:
    while True:
        # Read audio from the mic
        audio_chunk = stream_in.read(BLOCK_SHIFT, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize

        # Shift buffer and insert new data
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = audio_data

        # Process with DTLN
        input_tensor = torch.from_numpy(in_buffer).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            out_block, in_state1, in_state2 = model(input_tensor, in_state1, in_state2)
        
        out_block = out_block.numpy().squeeze()

        # Shift output buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT))
        out_buffer += out_block

        # Convert back to int16 for playback
        output_audio = (out_buffer[:BLOCK_SHIFT] * 32768.0).astype(np.int16).tobytes()

        # Play denoised audio through speakers
        stream_out.write(output_audio)

except KeyboardInterrupt:
    print("\n🛑 Stopping real-time noise suppression...")
    stream_in.stop_stream()
    stream_out.stop_stream()
    stream_in.close()
    stream_out.close()
    p.terminate()