import numpy as np
import matplotlib.pyplot as plt


# SRRC pulse generator

def generate_srrc_pulse(beta,Tsym=1.0,L=4,Nsym=8):

    """
    Generates Square-Root Raised Cosine (SRRC) pulse samples.
    """
    t = np.arange(-Nsym/2, Nsym/2 + 1/L, 1/L)
    p = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        if ti == 0.0:
            p[i] = (1.0 / np.sqrt(Tsym)) * (1.0 + beta * (4.0 / np.pi - 1.0))
        elif beta != 0 and abs(abs(ti) - Tsym / (4.0 * beta)) < 1e-8:
            p[i] = (beta / np.sqrt(2.0 * Tsym)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            term1 = np.sin(np.pi * (ti / Tsym) * (1.0 - beta))
            term2 = 4.0 * beta * (ti / Tsym) * np.cos(np.pi * (ti / Tsym) * (1.0 + beta))
            denom = np.pi * (ti / Tsym) * (1.0 - (4.0 * beta * (ti / Tsym))**2)
            p[i] = (1.0 / np.sqrt(Tsym)) * (term1 + term2) / denom
            
    # Normalize energy to unity
    p = p / np.sqrt(np.sum(p**2))
    return t, p


# Generate BPSK symbols

def generate_bpsk_symbols(num_bits):
    bits = np.random.randint(0,2,num_bits)

    # 1 - > -1
    # 0 - > 1

    symbols=np.where(bits==0,1,-1)

    return bits,symbols

def upsample_symbols(symbols,L):
    u=np.zeros(len(symbols)*L)

    u[::L] = symbols

    return u

def add_awgn(signal,snr_db):
    # Average signal power
    Ps=np.mean(signal**2)

    # Convert SNR dB to linear

    snr_linear = 10**(snr_db / 10)

    # Noise power
    Pn = Ps/snr_linear

    noise = np.sqrt(Pn)*np.random.randn(len(signal))

    received=signal+noise

    return received

# Complete Tx and Rx

def simulate_eye_signal(
        num_bits=1000,
        L=4,
        Tsym=1.0,
        Nsym=8,
        beta=0.4,
        snr_db=8
):

    bits,symbols=generate_bpsk_symbols(num_bits)

    u=upsample_symbols(symbols,L)

    t,p= generate_srrc_pulse(
        beta,
        Tsym,
        L,
        Nsym
    )



    tx=np.convolve(u,p,mode='full') # Tx 

    rx=add_awgn(tx,snr_db) # Rx 

    g = p[::-1] # Matched filter


    mf_output = np.convolve(rx,g,mode='full')



    N = len(p)
    single_delay = (N-1)//2

    total_delay = 2*single_delay

    # downsampling

    down_sampled = mf_output[total_delay::L]

    return {
        'bits':bits,
        'symbols':symbols,
        'upsampled':u,
        'pulse':p,
        'time':t,
        "tx":tx,
        "rx":rx,
        "matched_output":mf_output,
        "downsampled":down_sampled
    }


# EYE diagram

def plot_eye(
        signal,
        L=4,
        nSamples=None,
        nTraces=100,
        title="Eye Diagram"
):
    if nSamples is None:
        nSamples = 3*L
        required_samples = nSamples*nTraces

        signal = signal[:required_samples]

        time_axis = np.arange(nSamples) / L

        plt.figure(figsize=(8,5))

        for i in range(nTraces):


            start = i*nSamples
            end = start+nSamples

            trace=signal[start:end]

            if len(trace) == nSamples:
                plt.plot(time_axis,
                         trace)
                plt.axhline(
                        0,
                        linestyle='--'
                    )
                
                plt.axvline(
                    1,
                    linestyle='--'
                )
            
                plt.xlabel(
                    "Time (symbol periods)"
                )
            
                plt.ylabel(
                    "Amplitude"
                )
            
                plt.title(title)
            
                plt.grid(True)
            
                plt.show()


def main_eye_diagram():
    L=4
    beta=0.4
    snr_db = 8

    signals = simulate_eye_signal(
        num_bits=2000,
        L=L,
        beta=beta,
        snr_db=snr_db
    )

    plot_eye(
        signals["downsampled"],
        L=L,
        nSamples=3 * L,
        nTraces=100,
        title=f"Eye Diagram: β={beta}, SNR={snr_db} dB"

    )


# ============================================================
# 8. EFFECT OF SNR
# ============================================================

def compare_snr():

    L = 4
    beta = 0.4

    snr_values = [0, 4, 8]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    for ax, snr_db in zip(axes, snr_values):

        signals = simulate_eye_signal(
            num_bits=2000,
            L=L,
            beta=beta,
            snr_db=snr_db
        )

        signal = signals["downsampled"]

        nSamples = 3 * L
        nTraces = 100

        time_axis = np.arange(nSamples) / L

        for i in range(nTraces):

            start = i * nSamples
            end = start + nSamples

            trace = signal[start:end]

            if len(trace) == nSamples:

                ax.plot(
                    time_axis,
                    trace
                )

        ax.axhline(
            0,
            linestyle='--'
        )

        ax.set_title(
            f"SNR = {snr_db} dB"
        )

        ax.set_xlabel(
            "Time (symbol periods)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(True)

    fig.suptitle(
        "Effect of SNR on Eye Diagram (β = 0.4)"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. EFFECT OF ROLL-OFF FACTOR
# ============================================================

def compare_rolloff():

    L = 4
    snr_db = 4

    beta_values = [0.1, 0.5, 0.9]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    for ax, beta in zip(
        axes,
        beta_values
    ):

        signals = simulate_eye_signal(
            num_bits=2000,
            L=L,
            beta=beta,
            snr_db=snr_db
        )

        signal = signals["downsampled"]

        nSamples = 3 * L
        nTraces = 100

        time_axis = np.arange(nSamples) / L

        for i in range(nTraces):

            start = i * nSamples
            end = start + nSamples

            trace = signal[start:end]

            if len(trace) == nSamples:

                ax.plot(
                    time_axis,
                    trace
                )

        ax.axhline(
            0,
            linestyle='--'
        )

        ax.set_title(
            f"β = {beta}"
        )

        ax.set_xlabel(
            "Time (symbol periods)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(True)

    fig.suptitle(
        "Effect of Roll-off Factor on Eye Diagram (SNR = 4 dB)"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # Basic eye diagram
    main_eye_diagram()

    # Effect of SNR
    compare_snr()

    # Effect of roll-off factor
    compare_rolloff()