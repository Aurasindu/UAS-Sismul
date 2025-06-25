import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Parameter sinyal
fs = 48000  # sample rate
duration = 2  # detik
t = np.arange(0, duration, 1/fs)
signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # sinyal 440Hz

# Simpan sinyal original
sf.write("original.wav", signal, fs)

# Seed = 1103 (11 Maret)
np.random.seed(1103)
watermark = np.random.choice([-1, 1], size=len(signal))

# Embed dengan dua bobot berbeda
alpha_weak = 0.01
alpha_strong = 0.05
wm_weak = signal + alpha_weak * watermark
wm_strong = signal + alpha_strong * watermark

# Simpan dua versi audio
sf.write("wm_weak.wav", wm_weak, fs)
sf.write("wm_strong.wav", wm_strong, fs)

plt.figure(figsize=(12, 4))
plt.plot(t[:1000], signal[:1000], label="Original")
plt.plot(t[:1000], wm_weak[:1000], label="Alpha 0.01")
plt.plot(t[:1000], wm_strong[:1000], label="Alpha 0.05")
plt.title("Waveform: Original vs Watermarked")
plt.legend()
plt.tight_layout()
plt.show()

def detect_watermark(embedded_signal, watermark):
    correlation = np.dot(embedded_signal, watermark) / len(watermark)
    return correlation

print("Deteksi watermark:")
print("Alpha 0.01 →", detect_watermark(wm_weak, watermark))
print("Alpha 0.05 →", detect_watermark(wm_strong, watermark))