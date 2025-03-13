import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define bi-Gaussian function
def bi_gaussian(x, A, x0, sigma_L, sigma_R):
    return np.where(
        x < x0,
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_L ** 2)),
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_R ** 2))
    )

# Generate synthetic data with multiple peaks
np.random.seed(42)
n_points = 2000
x_data = np.linspace(-50, 80, n_points)

# Define multiple peaks
peaks = [
    (10, -10, 2, 8),  # (A, x0, sigma_L, sigma_R)
    (8, -2, 1.5, 2.5),
    (12, 5, 2, 3),
    (9, 12, 2.5, 3.5),
    (7, 22, 2, 4),
    (9, 36, 4, 6),
]

y_true = np.zeros_like(x_data)
for A, x0, sigma_L, sigma_R in peaks:
    y_true += bi_gaussian(x_data, A, x0, sigma_L, sigma_R)

# Add random noise
noise = np.random.normal(0, 0.5, size=n_points)
y_noisy = y_true + noise

# Save to CSV
csv_file = "synthetic_multi_bi_gaussian.csv"
df = pd.DataFrame({"x": x_data, "y": y_noisy})
df.to_csv(csv_file, index=False)

# Plot data
plt.scatter(x_data, y_noisy, label="Noisy Data", color="black", s=10)
plt.plot(x_data, y_true, label="True Bi-Gaussian Peaks", color="red", linestyle="dashed")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

print(f"Synthetic data with multiple peaks saved to {csv_file}")
