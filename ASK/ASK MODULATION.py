import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_bits = 10          # Number of bits to transmit
Fs = 1000              # Sampling frequency (Hz)
Fc = 100               # Carrier frequency (Hz)
A1 = 1                 # Amplitude for binary '1'
A0 = 0                 # Amplitude for binary '0'
Eb_N0_dB = 10          # SNR (dB)
T = 1 / Fs             # Sampling period
time = np.arange(0, num_bits, T)  # Time vector
bit_duration = Fs      # Number of samples per bit

# Generate Random Binary Data
bits = np.random.randint(0, 2, num_bits)

# ASK Modulation
ask_modulated = np.array([A1 if bit == 1 else A0 for bit in bits])  # Map bits to amplitudes
carrier = np.cos(2 * np.pi * Fc * time)                             # Carrier signal
ask_signal = np.repeat(ask_modulated, Fs) * carrier                # Modulated ASK signal

# Add AWGN (Noise)
def add_awgn(signal, snr_dB):
    snr = 10**(snr_dB / 10.0)  # Convert SNR from dB to linear scale
    signal_power = np.mean(signal**2)  # Calculate signal power
    noise_power = signal_power / snr   # Calculate noise power
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))  # Generate noise
    return signal + noise

ask_signal_noisy = add_awgn(ask_signal, Eb_N0_dB)

# Coherent ASK Demodulation
# Multiply noisy signal with synchronized carrier
received_signal = ask_signal_noisy * carrier

# Integrate over each bit duration
integrated_signal = np.array([
    np.sum(received_signal[i * bit_duration:(i + 1) * bit_duration])
    for i in range(num_bits)
])

# Comparator (Threshold Decision)
threshold = (A1 + A0) / 2

demodulated_bits = (integrated_signal > threshold).astype(int)

# Plot Results
plt.figure(figsize=(12, 9))

# 1. Original ASK Signal
plt.subplot(5, 1, 1)
plt.plot(time, ask_signal, label="ASK Signal")
plt.title("ASK Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 2. Noisy ASK Signal
plt.subplot(5, 1, 2)
plt.plot(time, ask_signal_noisy, label="Noisy ASK Signal", color="orange")
plt.title("Noisy ASK Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 3. Multiplied Signal (Carrier Recovery)
plt.subplot(5, 1, 3)
plt.plot(time, received_signal, label="Received Signal (Multiplied)", color="green")
plt.title("Received Signal After Carrier Multiplication")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 4. Integrated Signal
plt.subplot(5, 1, 4)
plt.stem(np.arange(num_bits), integrated_signal, label="Integrated Signal")
plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
plt.title("Integrated Signal and Threshold")
plt.xlabel("Bit Index")
plt.ylabel("Integrated Value")
plt.grid(True)
plt.legend()


# 5. Comparator Output
plt.subplot(5, 1, 5)
plt.step(np.arange(num_bits), bits, label="Original Bits", where="mid")
plt.step(np.arange(num_bits), demodulated_bits, label="Demodulated Bits", where="mid", linestyle="--", color="red")
plt.title("Comparator Output (Recovered Bits)")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
