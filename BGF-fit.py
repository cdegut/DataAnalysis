import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Load CSV data
csv_file = "synthetic_multi_bi_gaussian.csv"  # Change this to your actual file
df = pd.read_csv(csv_file)

x_data = df.iloc[:, 0].values
y_data = df.iloc[:, 1].values

# Exclude peak regions for baseline estimation
threshold = np.percentile(y_data, 50)  # Use median-based threshold
y_baseline_mask = y_data < threshold
x_baseline = x_data[y_baseline_mask]
y_baseline = y_data[y_baseline_mask]

# Perform polynomial baseline estimation
baseline_order = 3  # Adjust polynomial order as needed
baseline_coeffs = np.polyfit(x_baseline, y_baseline, baseline_order)
baseline_curve = np.polyval(baseline_coeffs, x_data)

#Filtering data
y_data_filtered = savgol_filter(y_data - baseline_curve, int(len(y_data)/20), 3)

# Detect peaks to estimate number of bi-Gaussian components
peaks, _ = find_peaks(y_data - baseline_curve, height=np.max(y_data - baseline_curve) * 0.3, distance=50)
num_peaks = len(peaks)

# Define multi-peak bi-Gaussian function
def multi_bi_gaussian(x, *params):
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        A, x0, sigma_L, sigma_R = params[i*4:(i+1)*4]
        y += np.where(
            x < x0,
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_L ** 2)),
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_R ** 2))
        )
    return y

# Initial guess for detected peaks
initial_params = []
for peak in peaks:
    A_guess = y_data[peak] - baseline_curve[peak]
    x0_guess = x_data[peak]
    sigma_L_guess = 2
    sigma_R_guess = 3
    initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])

# Fit the function
popt, _ = opt.curve_fit(multi_bi_gaussian, x_data, y_data - baseline_curve, p0=initial_params)

# Generate fitted curve
x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
y_fit = multi_bi_gaussian(x_fit, *popt) + np.polyval(baseline_coeffs, x_fit)

# Compute left start points (crossing the baseline)
left_start_points = []
for i in range(num_peaks):
    A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]
    start_L = x0_fit - 3 * sigma_L_fit  # Approximate start of peak on left
    left_start_points.append(start_L)
    print(f"Peak {i+1}: A = {A_fit:.3f}, x0 = {x0_fit:.3f}, sigma_L = {sigma_L_fit:.3f}, start_L = {start_L:.3f}")

# Define bi-Gaussian function for individual peaks
def bi_gaussian(x, A, x0, sigma_L, sigma_R):
    return np.where(
        x < x0,
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_L ** 2)),
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_R ** 2))
    )

# Plot data, baseline, fit, and left start points
plt.scatter(x_data, y_data, label="Raw Data", color="black", s=10)
plt.plot(x_data, baseline_curve, label="Estimated Baseline", color="green", linestyle="dashed")
plt.plot(x_fit, y_fit, label="Multi Bi-Gaussian Fit", color="red")

# Plot individual bi-Gaussian curves
for i in range(num_peaks):
    A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]
    y_individual_fit = bi_gaussian(x_fit, A_fit, x0_fit, sigma_L_fit, sigma_R_fit)
    plt.plot(x_fit, y_individual_fit, label=f"Peak {i+1}", linestyle="dotted")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

print(f"Detected {num_peaks} peaks.")
