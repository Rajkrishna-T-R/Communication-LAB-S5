import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
f=10
A=1
A_dc=1
Fs=1000

phi=0
w=2*np.pi*f
t=np.arange(0,1,1/Fs)
N=len(t)
x_sin=A_dc+A*np.sin(w*t+phi)

x_min=np.min(x_sin)

x_max=np.max(x_sin)


bits=np.arange(2,9,1)
x_qtz=np.zeros(np.size(t))
P_signal=np.zeros(len(bits))

P_noise = np.zeros(len(bits))

SQNR_actual=np.zeros(len(bits))

SQNR_dB=np.zeros(len(bits))

x_qtz_demo = None
x_noise_demo= None
demo_bit = 3

# for plot for each bits
fig,axes = plt.subplots(len(bits),1,figsize=(10,14),sharex=True)



for i,b in enumerate(bits):
    
    L=2**b

    delta=((x_max-x_min)/(L))

    x_qtz=np.round((x_sin-x_min)/delta)*delta+x_min+delta/2

    x_noise=x_sin-x_qtz

    P_signal[i]=np.mean(np.square(x_sin))
    P_noise[i]=np.mean(np.square(x_noise))

    

    SQNR_actual[i]=P_signal[i]/P_noise[i]



    SQNR_dB[i]=10*np.log10(SQNR_actual[i])


    if b==demo_bit:
        x_qtz_demo=x_qtz
        x_noise_demo=x_noise

    axes[i].plot(t[:200],x_sin[:200],label="Original Signal",color="Blue",linewidth=1.5)
    axes[i].plot(t[:200],x_qtz[:200],label=f"Quantized ({b} bits)",color="red",linestyle='--')
    axes[i].set_ylabel("Amplitude")
    axes[i].set_title(f"bits used for quantization={b} (L={L} levels)")
    axes[i].legend(loc="upper right")
    axes[i].grid(True)
    
print(SQNR_dB)

fig.suptitle("Original vs Quantized waveforms across bit depths",fontsize=14)
fig.tight_layout()


fig2,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,10))


# Subplot 1: Signal vs Quantized Signal
ax1.plot(t[:100], x_sin[:100], label="Original Signal", color="b")
ax1.plot(t[:100], x_qtz_demo[:100], label=f"Quantized ({demo_bit} bits)", color="r", linestyle="--")
ax1.set_title(f"Original vs Quantized Signal ({demo_bit} bits)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True)

# Subplot 2: Quantization Error
ax2.plot(t[:100], x_noise_demo[:100], color="g")
ax2.set_title(f"Quantization Error Waveform ({demo_bit} bits)")
ax2.set_ylabel("Error")
ax2.grid(True)

# Subplot 3: SQNR vs Bits
ax3.stem(bits, SQNR_dB)
ax3.set_title("SQNR vs Bits")
ax3.set_xlabel("Bits (b)")
ax3.set_ylabel("SQNR (dB)")
ax3.grid(True)

plt.tight_layout()
plt.show()




