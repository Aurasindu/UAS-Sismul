import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load gambar selfie dan convert ke grayscale
img = cv2.imread("nomor-1/selfie.jpeg", cv2.IMREAD_GRAYSCALE)

# 2. Ambil satu blok 8x8 pertama
block = img[0:8, 0:8].astype(np.float32) - 128

# 3. DCT (Discrete Cosine Transform)
dct = cv2.dct(block)

# 4. Kuantisasi (pakai matrix standar JPEG)
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
quantized = np.round(dct / Q)

# 5. Dekompresi: dekuantisasi + inverse DCT
dequantized = quantized * Q
idct = cv2.idct(dequantized) + 128

# 6. Tampilkan perbandingan blok asli dan hasil decoding
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(block + 128, cmap='gray')
plt.title("Original Block")
plt.subplot(1, 2, 2)
plt.imshow(idct, cmap='gray')
plt.title("Decoded Block")
plt.show()
