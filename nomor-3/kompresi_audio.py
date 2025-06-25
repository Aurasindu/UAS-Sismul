from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Load file WAV
audio = AudioSegment.from_wav("nomor-3/sismul.wav")

# Kompres ke MP3
audio.export("compressed.mp3", format="mp3", bitrate="128k")

# Load ulang buat analisis
y1, sr1 = librosa.load("nomor-3/sismul.wav", sr=None)
y2, sr2 = librosa.load("compressed.mp3", sr=None)

# Tampilkan waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y1, sr=sr1, alpha=0.5, label='Original')
librosa.display.waveshow(y2, sr=sr2, alpha=0.5, label='Compressed')
plt.legend()
plt.title("Perbandingan Waveform Audio")
plt.tight_layout()
plt.show()

# Tampilkan ukuran file
size_wav = round(os.path.getsize("nomor-3/sismul.wav") / 1024, 2)
size_mp3 = round(os.path.getsize("compressed.mp3") / 1024, 2)
print(f"Ukuran WAV: {size_wav} KB")
print(f"Ukuran MP3: {size_mp3} KB")
