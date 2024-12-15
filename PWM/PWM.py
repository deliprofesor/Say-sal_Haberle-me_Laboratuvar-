import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parametreler
frequency = 50  # PWM Frekansı (Hz)
duty_cycle = 50  # Duty Cycle (%)
duration = 0.05  # Süre (s)
fs = 10000  # Örnekleme frekansı (Hz)
cutoff = 100  # Filtreleme frekansı (Hz)

# Zaman aralığı
t = np.linspace(0, duration, 1000)

# 1. Adım: Giriş işareti (Analog sinüs dalgası)
input_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz'lik analog sinüs dalgası

# 2. Adım: PWM Modülasyonu (Modüle Edici İşaret - Dijital PWM sinyali)
pwm_signal = np.where((t % (1 / frequency)) < (1 / frequency) * (duty_cycle / 100), 1, 0)

# 3. Adım: Grafik - Giriş İşareti, PWM Modülasyonu ve Çıkış
plt.figure(figsize=(10, 8))

# Giriş işareti (Analog sinüs dalgası)
plt.subplot(3, 1, 1)
plt.plot(t, input_signal, label='Giriş İşareti (Analog Sinüs Dalga)', color='green')
plt.title('Adım 1: Giriş İşareti (Analog Sinüs Dalga)')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.grid(True)
plt.legend()

# PWM sinyali (Modüle Edici İşaret)
plt.subplot(3, 1, 2)
plt.plot(t, pwm_signal, label=f'Modüle Edici İşaret (PWM, Duty Cycle: {duty_cycle}%)', color='blue')
plt.title('Adım 2: Modüle Edici İşaret (PWM)')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.grid(True)
plt.legend()

# PWM çıkışı (Analog sinyalle modüle edilmiş PWM)
output_signal = input_signal * pwm_signal  # Analog sinyal ile PWM modülasyonu
plt.subplot(3, 1, 3)
plt.plot(t, output_signal, label='Çıkış İşareti (PWM Modülasyonu)', color='orange')
plt.title('Adım 3: Çıkış İşareti (PWM Modülasyonu)')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Adım 4: Demodülasyon (PWM sinyalini çözme - Düşük Geçiş Filtresi)
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# PWM sinyalini demodülasyon işlemi (filtreleme)
demodulated_signal = low_pass_filter(output_signal, cutoff, fs)

# Grafik - Demodülasyon Sonucu
plt.figure(figsize=(10, 6))

# Demodülasyona hazırlanan PWM çıkışı
plt.subplot(2, 1, 1)
plt.plot(t, output_signal, label='PWM Çıkışı (Modüle Edilmiş)', color='orange')
plt.title('Adım 4: PWM Çıkışı (Modüle Edilmiş)')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.grid(True)
plt.legend()

# Demodüle edilmiş sinyal (Çıkış)
plt.subplot(2, 1, 2)
plt.plot(t, demodulated_signal, label='Demodüle Edilmiş Sinyal (Analog)', color='purple')
plt.title('Adım 5: Demodülasyon (Rekonstrüksiyon)')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
