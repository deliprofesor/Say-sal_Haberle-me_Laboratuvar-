import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parametreler
num_bits = 10           # Gönderilecek bit sayısı (örnek az tutuldu)
Fs = 1000               # Örnekleme frekansı (Hz)
Fc = 100                # Taşıyıcı frekansı (Hz)
A1 = 1                  # 1 için genlik
A0 = 0                  # 0 için genlik
Eb_N0_dB = 10           # SNR (dB)
T = 1 / Fs              # Örnekleme süresi
time = np.arange(0, num_bits, 1/Fs)  # Zaman vektörü

# Rastgele Binary Veri Üretme
bits = np.random.randint(0, 2, num_bits)

# ASK Modülasyonu
ask_modulated = np.array([A1 if bit == 1 else A0 for bit in bits])  # 1 için A1, 0 için A0
ask_signal = np.tile(ask_modulated, Fs) * np.cos(2 * np.pi * Fc * time)

# Gürültü Ekleme
def add_awgn(signal, snr_dB):
    snr = 10**(snr_dB/10.0)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

ask_signal_noisy = add_awgn(ask_signal, Eb_N0_dB)

# Noncoherent Envelope Detector
envelope = np.abs(ask_signal_noisy)  # Zarf dedektörü çıkışı

# Low-Pass Filter
def low_pass_filter(signal, cutoff, Fs, order=6):
    nyquist = Fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

cutoff_freq = Fc / 2  # Alçak geçiren filtre kesim frekansı
lowpass_output = low_pass_filter(envelope, cutoff_freq, Fs)

# Comparator (Binarization)
threshold = (A1 + A0) / 2
comparator_output = (lowpass_output > threshold).astype(int)

# Çizim
plt.figure(figsize=(12, 8))

# 1. ASK Sinyali
plt.subplot(4, 1, 1)
plt.plot(time, ask_signal_noisy, label="Giriş Sinyali (Noisy ASK)")
plt.title("Giriş Sinyali")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid(True)
plt.legend()

# 2. Envelope Detector Çıkışı
plt.subplot(4, 1, 2)
plt.plot(time, envelope, label="Zarf Dedektörü Çıkışı", color="orange")
plt.title("Zarf Dedektörü Çıkışı (Envelope)")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid(True)
plt.legend()

# 3. Low-Pass Filter Çıkışı
plt.subplot(4, 1, 3)
plt.plot(time, lowpass_output, label="Low-Pass Çıkışı", color="green")
plt.title("Low-Pass Filter Çıkışı")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid(True)
plt.legend()

# 4. Comparator Çıkışı
plt.subplot(4, 1, 4)
plt.step(np.arange(num_bits), bits, label="Orijinal Veri", where="mid")
plt.step(np.arange(num_bits), comparator_output[::Fs], label="Comparator Çıkışı", where="mid", linestyle="--", color="red")
plt.title("Comparator Çıkışı")
plt.xlabel("Bit Zamanı")
plt.ylabel("Bit Değeri")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
