import numpy as np
import matplotlib.pyplot as plt

# FSK Modülasyonu
def fsk_modulation(bits, f1, f2, sampling_frequency):
    time = np.arange(0, len(bits), 1 / sampling_frequency)
    signal = np.zeros(len(time))

    for i, bit in enumerate(bits):
        # Bit 0 için f1, bit 1 için f2 frekanslarını kullan
        if bit == 0:
            carrier = np.cos(2 * np.pi * f1 * time[i * sampling_frequency:(i + 1) * sampling_frequency])
        else:
            carrier = np.cos(2 * np.pi * f2 * time[i * sampling_frequency:(i + 1) * sampling_frequency])
        signal[i * sampling_frequency:(i + 1) * sampling_frequency] = carrier

    return signal, time

# FSK Demodülasyonu
def fsk_demodulation(received_signal, f1, f2, sampling_frequency, num_bits):
    time = np.arange(0, num_bits, 1 / sampling_frequency)
    demodulated_bits = []

    for i in range(num_bits):
        segment = received_signal[i * sampling_frequency:(i + 1) * sampling_frequency]
        correlation_f1 = np.sum(segment * np.cos(2 * np.pi * f1 * time[i * sampling_frequency:(i + 1) * sampling_frequency]))
        correlation_f2 = np.sum(segment * np.cos(2 * np.pi * f2 * time[i * sampling_frequency:(i + 1) * sampling_frequency]))
        if correlation_f1 > correlation_f2:
            demodulated_bits.append(0)
        else:
            demodulated_bits.append(1)

    return np.array(demodulated_bits)

# Parametreler
num_bits = 8
f1 = 5  # İlk taşıyıcı frekansı
f2 = 10  # İkinci taşıyıcı frekansı
sampling_frequency = 1000  # Yüksek örnekleme frekansı
bits = np.random.randint(0, 2, num_bits)

# FSK Modülasyonu
fsk_signal, time_fsk = fsk_modulation(bits, f1, f2, sampling_frequency)

# FSK Demodülasyonu
demodulated_bits_fsk = fsk_demodulation(fsk_signal, f1, f2, sampling_frequency, num_bits)

# Grafik 1: Dijital Sinyal
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.step(range(num_bits), bits, where='post', label='Dijital Sinyal (Bits)', color='royalblue')
plt.title('Dijital Sinyal (Bits)', fontsize=14)
plt.xlabel('Bit Index', fontsize=12)
plt.ylabel('Bit Değeri', fontsize=12)
plt.grid(True)
plt.legend()

# Grafik 2: FSK Modülasyonu
plt.subplot(4, 1, 2)
plt.plot(time_fsk, fsk_signal, label='FSK Modüle Edilmiş Sinyal', color='darkorange')
plt.title('FSK Modüle Edilmiş Sinyal', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Genlik', fontsize=12)
plt.grid(True)
plt.legend()

# Grafik 3: Çıkış (Demodülasyon Sonrası)
plt.subplot(4, 1, 3)
plt.step(range(num_bits), demodulated_bits_fsk, where='post', label='Demodüle Edilmiş Bits', linestyle='--', color='green')
plt.title('Demodüle Edilmiş Bits (FSK)', fontsize=14)
plt.xlabel('Bit Index', fontsize=12)
plt.ylabel('Bit Değeri', fontsize=12)
plt.grid(True)
plt.legend()

# Grafik 4: FSK Modülasyonu ve Dijital Sinyalin Karşılaştırması
time_for_bits = np.linspace(0, num_bits, num_bits * sampling_frequency)
plt.subplot(4, 1, 4)
plt.plot(time_fsk, fsk_signal, label='FSK Modüle Edilmiş Sinyal', color='darkorange')
plt.step(time_for_bits, np.repeat(bits, sampling_frequency), where='post', label='Dijital Sinyal (Bits)', color='royalblue', linestyle='--')
plt.title('FSK ve Dijital Sinyalin Karşılaştırması', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Genlik / Bit Değeri', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
