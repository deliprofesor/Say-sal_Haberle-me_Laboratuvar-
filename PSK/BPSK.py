# Python Simulation for PSK (BPSK) and QPSK Modulation and Demodulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.special import erfc

# Parameters
num_bits = 100            # Number of bits to transmit
Fs = 1000                # Sampling frequency (Hz)
Fc = 100                 # Carrier frequency (Hz)
Eb_N0_dB = 10            # Energy per bit to noise power spectral density ratio (dB)

# Generate Random Binary Data
bits = np.random.randint(0, 2, num_bits)

# BPSK Modulation
bpsk_modulated = 2*bits - 1  # Map 0 -> -1, 1 -> +1
T = 1 / Fs                  # Sampling period
time = np.arange(0, num_bits, 1/Fs)  # Time vector
bpsk_signal = bpsk_modulated.repeat(Fs) * np.cos(2 * np.pi * Fc * time)

# Add Noise to BPSK Signal
def add_awgn(signal, snr_dB):
    snr = 10**(snr_dB/10.0)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

bpsk_signal_noisy = add_awgn(bpsk_signal, Eb_N0_dB)

# BPSK Demodulation
received_bpsk = bpsk_signal_noisy * np.cos(2 * np.pi * Fc * time)
received_bpsk_integrated = np.array([np.sum(received_bpsk[i*Fs:(i+1)*Fs]) for i in range(num_bits)])
bpsk_demodulated = (received_bpsk_integrated > 0).astype(int)

# QPSK Modulation
symbols = bits.reshape(-1, 2)  # Group bits into pairs
I = 2*symbols[:, 0] - 1       # In-phase component
Q = 2*symbols[:, 1] - 1       # Quadrature component
qpsk_symbols = (1/np.sqrt(2)) * (I + 1j*Q)
qpsk_time = np.arange(0, len(symbols), 1/Fs)
qpsk_signal = (np.real(qpsk_symbols.repeat(Fs)) * np.cos(2 * np.pi * Fc * qpsk_time) -
               np.imag(qpsk_symbols.repeat(Fs)) * np.sin(2 * np.pi * Fc * qpsk_time))

# Add Noise to QPSK Signal
qpsk_signal_noisy = add_awgn(qpsk_signal, Eb_N0_dB)

# QPSK Demodulation
received_I = qpsk_signal_noisy * np.cos(2 * np.pi * Fc * qpsk_time)
received_Q = qpsk_signal_noisy * np.sin(2 * np.pi * Fc * qpsk_time)
received_I_integrated = np.array([np.sum(received_I[i*Fs:(i+1)*Fs]) for i in range(len(symbols))])
received_Q_integrated = np.array([np.sum(received_Q[i*Fs:(i+1)*Fs]) for i in range(len(symbols))])
demodulated_I = (received_I_integrated > 0).astype(int)
demodulated_Q = (received_Q_integrated > 0).astype(int)
demodulated_bits_qpsk = np.column_stack((demodulated_I, demodulated_Q)).ravel()

# Calculate BER
bpsk_ber = np.sum(bits != bpsk_demodulated) / num_bits
qpsk_ber = np.sum(bits != demodulated_bits_qpsk) / num_bits
print(f"BPSK BER: {bpsk_ber:.4f}")
print(f"QPSK BER: {qpsk_ber:.4f}")

# Plot Signals
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, bpsk_signal)
plt.title("BPSK Modulated Signal")

plt.subplot(3, 1, 2)
plt.plot(time, bpsk_signal_noisy)
plt.title("Noisy BPSK Signal")

plt.subplot(3, 1, 3)
plt.stem(bits, label="Original")
plt.stem(bpsk_demodulated, linefmt="--r", markerfmt="ro", label="Demodulated")
plt.legend()
plt.title("BPSK Demodulation")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(qpsk_time, qpsk_signal)
plt.title("QPSK Modulated Signal")

plt.subplot(3, 1, 2)
plt.plot(qpsk_time, qpsk_signal_noisy)
plt.title("Noisy QPSK Signal")

plt.subplot(3, 1, 3)
plt.stem(bits, label="Original")
plt.stem(demodulated_bits_qpsk, linefmt="--r", markerfmt="ro", label="Demodulated")
plt.legend()
plt.title("QPSK Demodulation")
plt.tight_layout()
plt.show()
