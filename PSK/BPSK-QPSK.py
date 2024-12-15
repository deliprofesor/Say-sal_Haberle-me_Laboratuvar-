import numpy as np
import matplotlib.pyplot as plt

# BPSK Modulation and Demodulation
def bpsk_modulation(bits, carrier_frequency, sampling_frequency):
    time = np.arange(0, len(bits), 1 / sampling_frequency)
    carrier = np.cos(2 * np.pi * carrier_frequency * time)
    modulated_signal = np.repeat(2 * bits - 1, sampling_frequency) * carrier
    return modulated_signal, carrier, time

# QPSK Modulation and Demodulation
def qpsk_modulation(bits, carrier_frequency, sampling_frequency):
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # Make the length even

    I = 2 * bits[0::2] - 1  # In-phase bits
    Q = 2 * bits[1::2] - 1  # Quadrature bits

    time = np.arange(0, len(I), 1 / sampling_frequency)
    carrier_I = np.cos(2 * np.pi * carrier_frequency * time)
    carrier_Q = np.sin(2 * np.pi * carrier_frequency * time)

    modulated_signal = (
        np.repeat(I, sampling_frequency) * carrier_I + 
        np.repeat(Q, sampling_frequency) * carrier_Q
    )
    return modulated_signal, carrier_I, carrier_Q, time

# Parameters
num_bits = 8
carrier_frequency = 10
sampling_frequency = 1000  # Increased sampling frequency for better resolution
bits = np.random.randint(0, 2, num_bits)

# BPSK Modulation
bpsk_signal, carrier_bpsk, time_bpsk = bpsk_modulation(bits, carrier_frequency, sampling_frequency)

# QPSK Modulation
qpsk_signal, carrier_I_qpsk, carrier_Q_qpsk, time_qpsk = qpsk_modulation(bits, carrier_frequency, sampling_frequency)

# Plot BPSK Modulation Results (step by step)
plt.figure(figsize=(12, 9))

# Original Bits for BPSK
plt.subplot(3, 1, 1)
plt.step(range(num_bits), bits, where='post', label='Original Bits (BPSK)', color='royalblue')
plt.title('Original Bits (BPSK)', fontsize=14)
plt.xlabel('Bit Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True)
plt.legend()

# Carrier Signal for BPSK
plt.subplot(3, 1, 2)
plt.plot(time_bpsk, carrier_bpsk, label='Carrier Signal (BPSK)', color='green')
plt.title('Carrier Signal (BPSK)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.legend()

# BPSK Modulated Signal
plt.subplot(3, 1, 3)
plt.plot(time_bpsk, bpsk_signal, label='BPSK Signal', color='darkorange')
plt.title('BPSK Modulated Signal', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot QPSK Modulation Results (step by step)
plt.figure(figsize=(12, 9))

# Original Bits for QPSK
plt.subplot(4, 1, 1)
plt.step(range(num_bits), bits, where='post', label='Original Bits (QPSK)', color='royalblue')
plt.title('Original Bits (QPSK)', fontsize=14)
plt.xlabel('Bit Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True)
plt.legend()

# Carrier I for QPSK
plt.subplot(4, 1, 2)
plt.plot(time_qpsk, carrier_I_qpsk, label='Carrier I (QPSK)', color='green')
plt.title('Carrier I (QPSK)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.legend()

# Carrier Q for QPSK
plt.subplot(4, 1, 3)
plt.plot(time_qpsk, carrier_Q_qpsk, label='Carrier Q (QPSK)', color='purple')
plt.title('Carrier Q (QPSK)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.legend()

# QPSK Modulated Signal
plt.subplot(4, 1, 4)
plt.plot(time_qpsk, qpsk_signal, label='QPSK Modulated Signal', color='darkorange')
plt.title('QPSK Modulated Signal', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
