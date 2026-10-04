# Pulse shaping and Matched Filter

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def generate_srrc_pulse(beta, Tsym, L, Nsym):
 # L => samples per symbol 
    """
    Generate Square-Root raised Cosine (SRRC) pulse sample.

    """
    t=np.arange(-Nsym/2,Nsym/2+1/L,1/L)
    p=np.zeros_like(t)

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

    p = p/np.sqrt(np.sum(p**2))
    return t,p


def simulate_transceiver(num_bits=1000,L=4,Tsym=0.1,Nsym=8,beta=0.5,snr_db=10.0):

        # Step 1 :Bit Generation
        bits = np.random.randint(0,2,num_bits)

        #Step 2 : BPSK Mapping 
        # 1 - > 0  , 0 - > -1
        a=np.where(bits==0,1,-1)


        # Step 3 upsampling

        u=np.zeros(num_bits*L)
        u[::L] = a

        # Step 4 SRRC pulse
        t_pulse, p = generate_srrc_pulse(beta,Tsym,L,Nsym)

        # step 5 Transmit Filtering (Convolution)

        s=np.convolve(u,p,mode='full')

        # step 6 add AWGN channel

        Ps = np.mean(s**2)
        snr_lin = 10.0**(snr_db/10.0)
        Pn = Ps / snr_lin
        noise = np.sqrt(Pn) * np.random.randn(len(s))

        r = s + noise

        #step 7 Matched filtering

        g = p[::-1] # Matched filter
        y=np.convolve(r,g,mode='full')

        # Step 8  Delay allignment and Downsampling
        N = len(p)
        total_delay=N-1
        # sample at symbol peaks
        y_sampled = y[total_delay: total_delay+num_bits*L:L]

        # step 9  # BPSK demapping
        bits_hat = np.where(y_sampled >= 0,0,1)

        # Step 10 : bit error calc
        ber = np.mean(bits != bits_hat)

        signals={
                'bits':bits,'a':a,'u':u,'t_pulse':t_pulse,'p':p,
                's':s,'r':r,'y':y,'y_sampled':y_sampled,'bits_hat':bits_hat
        }

        return ber,signals




# ==========================================
# 3. PLOTTING TIME-DOMAIN STAGES
# ==========================================
def plot_simulation_stages():
    L = 4
    Nsym = 8
    beta = 0.5
    snr_db = 12.0
    num_bits = 20
    
    _, sigs = simulate_transceiver(num_bits=num_bits, L=L, Tsym=1.0, Nsym=Nsym, beta=beta, snr_db=snr_db)
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle(f"Pulse Shaping Simulation Stages (SNR = {snr_db} dB, $\\beta$ = {beta})", fontsize=14)
    
    # 1. Generated Bits
    axes[0, 0].stem(range(num_bits), sigs['bits'])
    axes[0, 0].set_title("1. Generated Binary Message")
    axes[0, 0].grid(True)
    
    # 2. Upsampled Baseband
    axes[0, 1].stem(range(len(sigs['u'][:num_bits*L])), sigs['u'][:num_bits*L])
    axes[0, 1].set_title("2. Upsampled Baseband Signal u[n]")
    axes[0, 1].grid(True)
    
    # 3. SRRC Pulse
    axes[1, 0].plot(sigs['t_pulse'], sigs['p'], 'r-', lw=2)
    axes[1, 0].set_title("3. SRRC Impulse Response p[n]")
    axes[1, 0].grid(True)
    
    # 4. Transmitted Baseband Signal
    axes[1, 1].plot(sigs['s'][:num_bits*L], 'g-')
    axes[1, 1].set_title("4. Transmitted Shaped Signal s[n]")
    axes[1, 1].grid(True)
    
    # 5. Received Signal with Noise
    axes[2, 0].plot(sigs['r'][:num_bits*L], 'm-')
    axes[2, 0].set_title("5. Received Signal r[n] (AWGN)")
    axes[2, 0].grid(True)
    
    # 6. Matched Filter Output
    axes[2, 1].plot(sigs['y'][:num_bits*L + len(sigs['p'])], 'b-')
    axes[2, 1].set_title("6. Matched Filter Output y[n]")
    axes[2, 1].grid(True)
    
    # 7. Sampled Symbols
    axes[3, 0].stem(range(num_bits), sigs['y_sampled'])
    axes[3, 0].axhline(0, color='k', linestyle='--')
    axes[3, 0].set_title("7. Sampled Decision Symbols")
    axes[3, 0].grid(True)
    
    # 8. Detected Bits
    axes[3, 1].stem(range(num_bits), sigs['bits_hat'])
    axes[3, 1].set_title("8. Detected Binary Sequence")
    axes[3, 1].grid(True)
    
    plt.tight_layout()
    plt.show()



# ==========================================
# 4. ROLL-OFF FACTOR COMPARISON
# ==========================================
def plot_rolloff_comparisons():
    betas = [0.0, 0.25, 0.5, 1.0]
    L = 4
    Nsym = 8
    Tsym = 1.0
    
    plt.figure(figsize=(10, 4))
    for beta in betas:
        t_p, p = generate_srrc_pulse(beta, Tsym, L, Nsym)
        plt.plot(t_p, p, label=f'$\\beta = {beta}$')
        
    plt.title("SRRC Impulse Responses for Various Roll-off Factors ($\\beta$)")
    plt.xlabel("Time ($t / T_{sym}$)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # BER vs SNR performance

def plt_ber_curves():
    snr_db_range = np.arange(0,16,2)
    num_bits=50000
    betas = [0.25,0.5,1.0]

    plt.figure(figsize=(9,6))

    for beta in betas:
        ber_list=[]
        for snr in snr_db_range:
                ber,_ = simulate_transceiver(num_bits=num_bits,L=4,Tsym=1.0,Nsym=8,beta=beta,snr_db=snr)
                ber_list.append(ber)
        plt.semilogy(snr_db_range, ber_list, 'o--', label=f'Simulated ($\\beta={beta}$)')


    # Theoretical BPSK BER over AWGN: Q(sqrt(2 * Eb/N0)) = 0.5 * erfc(sqrt(Eb/N0))
    # Note: SNR per sample to Eb/N0 relationship: Eb/N0 = SNR_linear * L
    snr_lin_range = 10.0**(snr_db_range / 10.0)
    eb_n0_lin = snr_lin_range * 4.0  # L = 4
    theory_ber = 0.5 * erfc(np.sqrt(eb_n0_lin))
    
    plt.semilogy(snr_db_range, theory_ber, 'k-', lw=2, label='Theoretical BPSK')
    
    plt.title("BER vs. SNR for BPSK with SRRC Pulse Shaping")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.ylim([1e-5, 1])
    plt.legend()
    plt.grid(True, which='both')
    plt.show()
            
    # Run plot generation
if __name__ == "__main__":

    plot_simulation_stages()
    plot_rolloff_comparisons()
    plt_ber_curves()


                   