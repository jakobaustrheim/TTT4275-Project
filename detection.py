import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def load_data(filename):
    return scipy.io.loadmat(filename)


Sk_Gaussian = load_data('Datasets/T1_data_Sk_Gaussian.mat')['T1_data_Sk_Gaussian']



Sn_Gaussian = np.zeros(1024, dtype=complex)
comp = np.imag(0)
c = complex(0, 1)

#Computes the inverse Fourier Transform
for j in range(1024):
    for i in range(1024):
        comp += 1/np.sqrt(1024) * (Sk_Gaussian[i] * np.exp((c * 2 * np.pi * i * j)/ 1024))
    Sn_Gaussian[j] = comp
    comp = np.imag(0)

#Extract real and imaginary parts
real_part = np.real(Sn_Gaussian)
imag_part = np.imag(Sn_Gaussian)
#Plot data
#Extract real and imaginary parts
# Create histograms 
plt.figure(figsize=(12, 5))

# Histogram of Real Part
plt.subplot(1, 2, 1)
plt.hist(real_part, bins=50, color='b', alpha=0.7, edgecolor='black')
plt.xlabel("Real Part Values")
plt.ylabel("Frequency")
plt.title("Histogram of Real Part")



# Histogram of Imaginary Part
plt.subplot(1, 2, 2)
plt.hist(imag_part, bins=50, color='r', alpha=0.7, edgecolor='black')
plt.xlabel("Imaginary Part Values")
plt.ylabel("Frequency")
plt.title("Histogram of Imaginary Part")

# Show the plots
plt.tight_layout()
plt.show

def variance(data):
    var = 0
    for i in range(len(data)):
        var += data[i]**2
    return var/len(data)

#Compute the variance of the real and imaginary parts
real_var = variance(real_part)
imag_var = variance(imag_part)



def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

#Compute the autocorrelation of the real and imaginary parts
real_autocorr = autocorr(real_part)
imag_autocorr = autocorr(imag_part)

#Plot the autocorrelation

mean_sn = np.mean(imag_part)
print(mean_sn)

