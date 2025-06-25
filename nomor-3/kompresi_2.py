import scipy.fft
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Load audio
y_orig, sr_orig = sf.read("nomor-3/sismul.wav")
y_mp3, sr_mp3 = sf.read("compressed.mp3")

# FFT
fft_orig = np.abs(np.fft.fft(y_orig))[:10000]
fft_mp3 = np.abs(np.fft.fft(y_mp3))[:10000]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(fft_orig, label="Original WAV", alpha=0.7)
plt.plot(fft_mp3, label="Compressed MP3", alpha=0.7)
plt.title("Perbandingan Spektrum Frekuensi")
plt.xlabel("Frekuensi (bin)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
