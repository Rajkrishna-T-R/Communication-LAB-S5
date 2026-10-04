import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

N=100000 # Total number of bits
Eb=1.0
eb_no_db_range=np.arange(-2,11,1) #Eb/N0 range in dB

np.random.seed(42)
bits = np.random.randint(0,2,N)



# Group bits into pairs
b1=bits[0::2] # Even index
b2=bits[1::2] # Odd index

"""
# Gray code mapping
00 -> 1+j
01 -> -1+j
11 -> -1-j
10 -> -1+j

real component depends on b2 and imaginary component depends on b1
 0 -> 1
 1 -> -1
"""
I_symbols = np.sqrt(Eb)*(1-2*b1)
R_symbols = np.sqrt(Eb)*(1-2*b2)

symbols=R_symbols + 1j*I_symbols

snr_plot_list = [0,5,10]
fig,axes = plt.subplots(1,4,figsize=(16,4))

axes[0].scatter(np.real(symbols[:1000]),np.imag(symbols[:1000]),color='blue',alpha=0.6)
axes[0].set_title('Ideal Constellation')
axes[0].set_xlim([-2.5, 2.5])
axes[0].set_ylim([-2.5, 2.5])
axes[0].axhline(0, color='black', lw=0.5)
axes[0].axvline(0, color='black', lw=0.5)
axes[0].grid(True)

# plot the constellations for o,5,10dB

for idx,snr_db in enumerate(snr_plot_list):
    snr_lin=10**(snr_db/10)
    N0=Eb/snr_lin
    sigma=np.sqrt(N0/2.0)

    # complex additive white gaussian noise

    noise=sigma * (np.random.randn(len(symbols))+1j*np.random.randn(len(symbols)))
    y = symbols + noise
    ax = axes[idx + 1]
    ax.scatter(np.real(y[:1000]),np.imag(y[:1000]),color='red',alpha=0.3,s=10,label='Received')

    # Ideal constellation points for reference
    ideal_pts=[np.sqrt(Eb)*(1+1j),np.sqrt(Eb)*(-1+1j),np.sqrt(Eb)*(-1-1j),np.sqrt(Eb)*(1-1j)]
    ax.scatter(np.real(ideal_pts),np.imag(ideal_pts),color='black',marker='x',s=60,label='Ideal')

    ax.set_title(f'Received ({snr_db} dB)')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True)

plt.tight_layout()
plt.show()


simualted_ber=[]
theoretical_ber=[]

for snr_db in eb_no_db_range:
    snr_lin=10**(snr_db/10.0)
    N0 = Eb / snr_lin
    sigma = np.sqrt(N0/2.0)

    noise =sigma*(np.random.randn(len(symbols))+1j*np.random.randn(len(symbols)))
    y=symbols+noise

    # Detected bits
    det_b1 = np.where(np.imag(y)>= 0,0,1)
    det_b2 = np.where(np.real(y)>= 0,0,1)

    detected_bits=np.zeros(N,dtype=int)
    detected_bits[0::2]=det_b1
    detected_bits[1::2]=det_b2

    bit_errors=np.sum(detected_bits != bits)

    ber=bit_errors/N
    simualted_ber.append(ber)

    Prob_error=0.5*erfc(np.sqrt(snr_lin))
    theoretical_ber.append(Prob_error)



print(f"Theoretical BER:{theoretical_ber}")


print(f"\nSimulated BER:{simualted_ber}")



plt.figure(figsize=(5,6))
plt.semilogy(eb_no_db_range,theoretical_ber,'b-',label='Theoretical BER')
plt.semilogy(eb_no_db_range,simualted_ber,'ro--',label="Simulated BER")
plt.title(r'QPSK Bit Error Rate (BER) vs $E_b/N_0$ over AWGN Channel')
plt.xlabel(r'$E_b/N_0$ (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()