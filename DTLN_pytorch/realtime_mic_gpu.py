import os
import torch
import numpy as np
import pyaudio
from DTLN_model_copy import Pytorch_DTLN_stateful

# === CONFIGURATION ===
MODEL_PATH = "./pretrained/model.pth"  # Update this path if necessary
RATE = 16000   # Sample rate (matches DTLN's training)
BLOCK_LEN = 512  # Number of samples per frame (32ms per frame)
BLOCK_SHIFT = 128  # Frame shift (8ms per step)
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio

# === CHECK FOR GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# === LOAD MODEL ===
print(f"🎯 Loading model from: {MODEL_PATH}")
model = Pytorch_DTLN_stateful()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === INIT AUDIO STREAMS ===
p = pyaudio.PyAudio()

# Microphone input stream
stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                   frames_per_buffer=BLOCK_LEN, input_device_index=None)

# Speaker output stream
stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                    frames_per_buffer=BLOCK_LEN, output_device_index=None)

print("🎙️ Real-time noise suppression running... Speak into the mic!")

# === REAL-TIME PROCESSING ===
# Buffers
in_buffer = np.zeros((BLOCK_LEN,))
out_buffer = np.zeros((BLOCK_LEN,))

# Initialize LSTM states on GPU
in_state1 = (
    torch.zeros(2, 1, 128, dtype=torch.float32, device=device),  # Hidden state
    torch.zeros(2, 1, 128, dtype=torch.float32, device=device)   # Cell state
)
in_state2 = (
    torch.zeros(2, 1, 128, dtype=torch.float32, device=device),
    torch.zeros(2, 1, 128, dtype=torch.float32, device=device)
)

try:
    while True:
        # Read audio from the mic
        audio_chunk = stream_in.read(BLOCK_SHIFT, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize

        # Shift buffer and insert new data
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = audio_data

        # Process with DTLN
        input_tensor = torch.from_numpy(in_buffer).unsqueeze(0).to(torch.float32).to(device)

        with torch.no_grad():
            # Ensure states are correctly formatted and on GPU
            h1, c1 = in_state1[0].to(device), in_state1[1].to(device)
            h2, c2 = in_state2[0].to(device), in_state2[1].to(device)

            # Forward pass on GPU
            out_block, (h1, c1), (h2, c2) = model(input_tensor, (h1, c1), (h2, c2))

            # Ensure states remain contiguous and stay on GPU
            in_state1 = (h1.contiguous(), c1.contiguous())
            in_state2 = (h2.contiguous(), c2.contiguous())

        # Move output back to CPU for audio playback
        out_block = out_block.cpu().numpy().squeeze()

        # Shift output buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT,))
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
