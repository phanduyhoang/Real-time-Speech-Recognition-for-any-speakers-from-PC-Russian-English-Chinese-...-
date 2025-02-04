##Real-time speech recognition
For Russian: run display_noise_cancelling_russian.py
For English: run display_noise_cancelling_english_improved.py

## Installation

### Step 1: Install Python and Dependencies
Ensure you have **Python 3.7+** installed. Then, install the required libraries:

```bash
pip install vosk sounddevice numpy
```

### Step 2: Download the Russian Vosk Model
1. Visit [Vosk Models](https://alphacephei.com/vosk/models)
2. Download the small Russian model (recommended for real-time processing). Example: `vosk-model-small-ru-0.22`
3. Extract the model into your project directory.

### Step 3: Configure System Audio (Windows)
Since Python cannot directly capture system audio, we use **VB-Audio Virtual Cable** to route sound.

#### Step 3.1: Install VB-Audio Virtual Cable
1. Download from [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
2. Install and restart your PC.

#### Step 3.2: Set Virtual Cable as Default Audio Output
1. Open Windows Sound Settings (`Win + R → mmsys.cpl`)
2. In **Output**, set `Speakers (VB-Audio Virtual Cable)` as the **Default Device**.
3. Now Python will listen to system audio instead of physical speakers.

#### Step 3.3: Enable Playback to Physical Device (So You Can Still Hear Sound)
1. In Windows Sound Settings, go to **Recording → Cable Output → Right-click → Properties**.
2. Go to the **Listen** tab.
3. Enable **"Listen to this device"**.
4. Under **"Playback through this device"**, select your physical speakers/headphones.
5. Click **OK**.

Now, Python will capture system audio, and you will still hear it normally.
