import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Initialization and Data Generation
# ---------------------------------------------------------
N = 10**5  # Number of samples
roll_number = 59  # Theoretical mean (r)
variance = 1  # Theoretical variance (sigma^2)
std_dev = np.sqrt(variance)

# Generate independent Gaussian random variables X and Y
x = np.random.normal(roll_number, std_dev, N)
y = np.random.normal(roll_number, std_dev, N)

# ---------------------------------------------------------
# 2. Statistics Calculation (Empirical vs Theoretical)
# ---------------------------------------------------------
mean_x, var_x = np.mean(x), np.var(x)
mean_y, var_y = np.mean(y), np.var(y)

z = x + y
mean_z, var_z = np.mean(z), np.var(z)

print("=== STATISTICAL COMPARISON ===")
print(f"X -> Empirical Mean: {mean_x:.4f} (Theoretical: {roll_number}), Variance: {var_x:.4f} (Theoretical: 1)")
print(f"Y -> Empirical Mean: {mean_y:.4f} (Theoretical: {roll_number}), Variance: {var_y:.4f} (Theoretical: 1)")
print(f"Z -> Empirical Mean: {mean_z:.4f} (Theoretical: {2 * roll_number}), Variance: {var_z:.4f} (Theoretical: 2)\n")

# ---------------------------------------------------------
# 3. Empirical PDFs and CDFs Computation
# ---------------------------------------------------------
# Calculate histogram density (PDF) and bin edges
pdf_x, bin_edges_x = np.histogram(x, bins=100, density=True)
pdf_y, bin_edges_y = np.histogram(y, bins=100, density=True)

# Calculate empirical CDFs using cumulative sum over bin widths
dx_x = np.diff(bin_edges_x)
ecdf_x = np.cumsum(pdf_x * dx_x)

dx_y = np.diff(bin_edges_y)
ecdf_y = np.cumsum(pdf_y * dx_y)

# ---------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------
# Figure 1: Histograms / PDFs of X and Y
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Empirical PDFs of X and Y
ax1.hist(x, bins=100, density=True, alpha=0.6, label='Empirical PDF of X', color='blue')
ax1.hist(y, bins=100, density=True, alpha=0.6, label='Empirical PDF of Y', color='orange')
ax1.set_title("Empirical PDF of X and Y")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")
ax1.grid(True)
ax1.legend()

# Subplot 2: Empirical CDFs of X and Y
ax2.plot(bin_edges_x[1:], ecdf_x, label='Empirical CDF of X', color='blue', linewidth=2)
ax2.plot(bin_edges_y[1:], ecdf_y, label='Empirical CDF of Y', color='orange', linestyle='--', linewidth=2)
ax2.set_title("Empirical CDF of X and Y")
ax2.set_xlabel("Value")
ax2.set_ylabel("Cumulative Probability")
ax2.grid(True)
ax2.legend()

# Subplot 3: Empirical PDF of Z = X + Y
ax3.hist(z, bins=100, density=True, color='purple', alpha=0.7, label='Empirical PDF of Z = X + Y')
ax3.set_title(f"Histogram (PDF) of Z = X + Y\n(Mean ≈ {mean_z:.2f}, Var ≈ {var_z:.2f})")
ax3.set_xlabel("Value")
ax3.set_ylabel("Density")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()