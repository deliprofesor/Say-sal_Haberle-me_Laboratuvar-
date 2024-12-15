import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_bits = 10           # Number of bits to transmit (example kept small)
Fs = 1000               # Sampling frequency (Hz)
Fc = 100                # Carrier frequency (Hz)
A1 = 1                  # Amplitude for '1'
A0 = 0                  # Amplitude for '0'
Eb_N0_dB = 10           # SNR (dB)
T = 1 / Fs              # Sampling period
time = np.arange(0, num_bits, 1/Fs)  # Time vector
bit_duration = int(Fs)  # Number of samples per bit

# Generate Random Binary Data
bits = np.random.randint(0, 2, num_bits)

# ASK Modulation
ask_modulated = np.array([A1 if bit == 1 else A0 for bit in bits])  # Map 1 -> A1, 0 -> A0
carrier = np.cos(2 * np.pi * Fc * time)
ask_signal = np.tile(ask_modulated, Fs) * carrier

# Add Noise to the Signal
def add_awgn(signal, snr_dB):
    snr = 10**(snr_dB/10.0)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

ask_signal_noisy = add_awgn(ask_signal, Eb_N0_dB)

# Coherent Demodulation
# Multiply with synchronized carrier
received_signal = ask_signal_noisy * carrier

# Integrate over each bit period
integrated_signal = np.array([np.sum(received_signal[i*bit_duration:(i+1)*bit_duration]) for i in range(num_bits)])

# Comparator (Threshold Decision)
threshold = (A1 + A0) / 2
demodulated_bits = (integrated_signal > threshold).astype(int)

# Plot Results
plt.figure(figsize=(12, 8))

# 1. Input Noisy ASK Signal
plt.subplot(4, 1, 1)
plt.plot(time, ask_signal_noisy, label="Noisy ASK Signal")
plt.title("Input Signal (Noisy ASK)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 2. Synchronized Carrier Signal
plt.subplot(4, 1, 2)
plt.plot(time[:bit_duration*2], carrier[:bit_duration*2], label="Synchronized Carrier", color="orange")
plt.title("Synchronized Carrier Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 3. Multiplied Signal (Receiver Output)
plt.subplot(4, 1, 3)
plt.plot(time, received_signal, label="Received Signal (Multiplied)", color="green")
plt.title("Received Signal After Carrier Multiplication")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 4. Comparator Output
plt.subplot(4, 1, 4)
plt.step(np.arange(num_bits), bits, label="Original Bits", where="mid")
plt.step(np.arange(num_bits), demodulated_bits, label="Demodulated Bits", where="mid", linestyle="--", color="red")
plt.title("Comparator Output (Recovered Bits)")
plt.xlabel("Bit Time")
plt.ylabel("Bit Value")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
