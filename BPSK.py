import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


# --------------------------------------------------
# Parameters
# --------------------------------------------------

num_bits = 1_000_000
Eb = 1

SNR_array = np.array([0, 2, 4, 6, 8, 10])


# --------------------------------------------------
# Generate random bits
# --------------------------------------------------

rand_bit_array = np.random.randint(0, 2, num_bits)


# --------------------------------------------------
# BPSK mapping
# 0 -> +sqrt(Eb)
# 1 -> -sqrt(Eb)
# --------------------------------------------------

symbols = np.where(
    rand_bit_array == 0,
    np.sqrt(Eb),
    -np.sqrt(Eb)
)


# --------------------------------------------------
# Constellation plots
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(SNR_array) + 1,
    figsize=(16, 4)
)


# Ideal constellation
axes[0].scatter(
    np.real(symbols[:1000]),
    np.imag(symbols[:1000])
)

axes[0].set_xlim([-2.5, 2.5])
axes[0].set_ylim([-2.5, 2.5])
axes[0].axhline(0, lw=0.5)
axes[0].axvline(0, lw=0.5)
axes[0].grid(True)
axes[0].set_title("Ideal Constellation")


# --------------------------------------------------
# Generate noisy constellations
# --------------------------------------------------

for idx, SNR_dB in enumerate(SNR_array):

    SNR_linear = 10 ** (SNR_dB / 10)

    N0 = Eb / SNR_linear

    sigma = np.sqrt(N0 / 2)

    noise = sigma * (
        np.random.randn(num_bits)
        + 1j * np.random.randn(num_bits)
    )

    y = symbols + noise

    axes[idx + 1].scatter(
        np.real(y[:1000]),
        np.imag(y[:1000]),
        color="red",
        alpha=0.5,
        s=10
    )

    axes[idx + 1].set_xlim([-2.5, 2.5])
    axes[idx + 1].set_ylim([-2.5, 2.5])
    axes[idx + 1].axhline(0, lw=0.5)
    axes[idx + 1].axvline(0, lw=0.5)
    axes[idx + 1].grid(True)

    axes[idx + 1].set_title(
        f"SNR = {SNR_dB} dB"
    )


plt.tight_layout()
plt.show()


# --------------------------------------------------
# BER calculation
# --------------------------------------------------

Simulated_BER = []
Theoretical_BER = []


for SNR_dB in SNR_array:

    SNR_linear = 10 ** (SNR_dB / 10)

    N0 = Eb / SNR_linear

    sigma = np.sqrt(N0 / 2)

    # AWGN
    noise = sigma * (
        np.random.randn(num_bits)
        + 1j * np.random.randn(num_bits)
    )

    # Received signal
    y = symbols + noise


    # BPSK detection
    detected_bits = np.where(
        np.real(y) >= 0,
        0,
        1
    )


    # Number of errors
    bit_errors = np.sum(
        rand_bit_array != detected_bits
    )


    # Simulated BER
    ber = bit_errors / num_bits

    Simulated_BER.append(ber)


    # Theoretical BER
    theoretical = Q(
        np.sqrt(2 * SNR_linear)
    )

    Theoretical_BER.append(theoretical)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(f"Theoretical BER:{Theoretical_BER}")


print(f"\nSimulated BER:{Simulated_BER}")



      
# --------------------------------------------------
# BER plot
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.semilogy(
    SNR_array,
    Theoretical_BER,
    'o-',
    label='Theoretical BER'
)

plt.semilogy(
    SNR_array,
    Simulated_BER,
    's-',
    label='Simulated BER'
)

plt.xlabel(r'$E_b/N_0$ (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BPSK BER Performance over AWGN')

plt.grid(True, which='both')
plt.legend()



plt.show()