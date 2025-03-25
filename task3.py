import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def load_data(filename):
    return scipy.io.loadmat(filename)

x_H1 = load_data('Datasets/T3_data_x_H1.mat')['T3_data_x_H1']
x_H0 = load_data('Datasets/T3_data_x_H0.mat')['T3_data_x_H0']

sigma_w_values = load_data('Datasets/T3_data_sigma_w.mat')['w']
sigma_s_values = load_data('Datasets/T3_data_sigma_s.mat')['s_t']

sigma_w = 0
sigma_s = 0

# for i in range(len(sigma_w_values)):
#     sigma_w += 1/(len(sigma_w_values)) * (sigma_w_values[i]**2)

# for i in range(len(sigma_s_values)):
#     sigma_s += 1/(len(sigma_s_values)) * (sigma_s_values[i]**2)

sigma_w = np.var(sigma_w_values)
sigma_s = np.var(sigma_s_values)

print(sigma_w)

imag_H1 = np.imag(x_H1)
real_H1 = np.real(x_H1)

imag_H0 = np.imag(x_H0)
real_H0 = np.real(x_H0)

chi_H0 = np.zeros(1024, dtype= complex)
chi_H1 = np.zeros(1024, dtype= complex)

for i in range(len(x_H0)):
    chi_H0[i] = (1/sigma_w) * 2 *(imag_H0[i]**2 + real_H0[i]**2)

for i in range(len(x_H1)):
    chi_H1[i] = (1/(sigma_w + sigma_s)) * 2 * (imag_H1[i]**2 + real_H1[i]**2)

#

x = np.linspace(0, 10, 100)
H0_pdf = chi2.pdf(x, 2)
H1_pdf = chi2.pdf(x, 2)
bins = 100


# Plot the histograms
# plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(chi_H0, bins=50, color='b', alpha=0.7, edgecolor='black', density=True)
plt.xlabel("Chi_H0 Values")
plt.ylabel("Frequency")
plt.plot(x, H0_pdf)
plt.legend(["Chi-squared with two degrees of freedom", "Histogram"])
plt.title("Histogram of Chi_H0")

plt.subplot(1, 2, 2)
plt.hist(chi_H1, bins=50, color='r', alpha=0.7, edgecolor='black', density=True)
plt.plot(x, H0_pdf)
plt.legend(["Chi-squared with two degrees of freedom", "Histogram"])
plt.xlabel("Chi_H1 Values") 
plt.ylabel("Frequency")
plt.title("Histogram of Chi_H1")
plt.tight_layout()
plt.show()

