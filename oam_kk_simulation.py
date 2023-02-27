from numpy import *
from numpy.fft import *
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Parameters for the simulation
# Maximum topological charge
MAX_TC = 20
# Ratio of the carrier amplitude to the maximum signal
# amplitude. The minimum phase condition will be
# satisfied when AMP_CARRIER > 1
AMP_CARRIER = 1.05
# Relative phase between the signal and the carrier
REL_PHASE = 4 * pi / 5
# Number of the azimuthal sample points
AZI_SAMPLE_NUM = 41
# Parameter for the upsampling. The azimuthal sample
# rate will become (2*PADWIDTH)+1 times of the original.
# Zero if not do upsampling.
PADWIDTH = 10
# Fix the randomly generated OAM spectrum for reproducibility
RANDOM_SEED = 104832

SPEC_AMP = random.rand(MAX_TC)
SPEC_PHA = 2 * pi * random.rand(MAX_TC) - pi
TC_RANGE = arange(1, MAX_TC + 1)
THETA = linspace(0, 2 * pi, AZI_SAMPLE_NUM, False)
PADTIMES = 2 * int(PADWIDTH) + 1
SPEC_AMP /= sqrt((sum(SPEC_AMP ** 2)))

# Calculate azimuthal distribution for the given spectrum
signal_field = zeros(AZI_SAMPLE_NUM, dtype=complex)
for tc, amp, pha in zip(TC_RANGE, SPEC_AMP, SPEC_PHA):
    signal_field += amp * exp(1j * tc * THETA) * exp(1j * pha)
signal_field /= sqrt(AZI_SAMPLE_NUM)  # Normalize

# Generate azimuthal uniform carrier beam with the defined
# CSPR and relative phase
carrier_field = AMP_CARRIER * max(abs(signal_field)) * \
                ones_like(THETA, dtype=complex) * \
                exp(1j * REL_PHASE)

# Simulate the azimuthal intensity distribution of the carrier
# beam and the interference
carrier_int = abs(carrier_field) ** 2
interference_int = abs((signal_field + carrier_field)) ** 2
gamma = interference_int / carrier_int

# Upsampling by means of zero-padding
gamma_fft = fftshift(fft(gamma))
gamma_fft_extend = pad(gamma_fft, PADWIDTH * AZI_SAMPLE_NUM)
gamma_extend = ifft(ifftshift(gamma_fft_extend)) * PADTIMES

# Implement Kramers-Kronig relations
kai_real = 1 / 2 * log(gamma_extend)
kai_imag = imag(hilbert(real(kai_real)))
kai = kai_real + 1j * kai_imag
kai = kai[::PADTIMES]  # Downsample to original sample rate
ret_field = sqrt(carrier_int) * (exp(kai) - 1)
ret_field /= sqrt(sum(abs(ret_field) ** 2))  # Normalize

# Obtain the complex OAM spectrum by Fourier Transform
ret_spec = fft(ret_field)[1:MAX_TC + 1]
# Add a constant phase shift to match the phase
phase_match_index = argmax(abs(ret_field))
phase_diff = - angle(ret_field)[phase_match_index] \
             + angle(signal_field)[phase_match_index]
ret_spec = ret_spec * exp(1j * phase_diff)
ret_spec /= sqrt(sum(abs(ret_spec) ** 2))  # Normalize

# Calculate the overlap integral and CSPR
oi = abs(sum(ret_field * conj(signal_field))) ** 2
cspr = sum(carrier_int) / sum(abs(signal_field) ** 2)
cspr_dB = 10 * log10(cspr)

# Output the results
print('Overlap integral: {:.4f}'.format(oi))
print('Current CSPR: {:.2f} dB'.format(cspr_dB))

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6))
width = 0.25
style_ret = {'label': 'Retrieved', 'edgecolor': 'g',
             'facecolor': 'none', 'hatch': '///'}
style_ideal = {'label': 'Ground truth', 'edgecolor': 'b',
               'facecolor': 'none', 'hatch': '--'}
ax0.bar(TC_RANGE - 0.5 * width, abs(ret_spec), width, **style_ret)
ax0.bar(TC_RANGE + 0.5 * width, SPEC_AMP, width, **style_ideal)
ax0.legend(loc='upper right')
ax0.set_xticks(arange(0, MAX_TC + 1, 5))
ax0.set_ylabel('Amplitude', fontsize=13)
ax0.set_yticks([])
ax1.bar(TC_RANGE - 0.5 * width, angle(ret_spec), width, **style_ret)
ax1.bar(TC_RANGE + 0.5 * width, SPEC_PHA, width, **style_ideal)
ax1.set_xlabel('OAM mode index', fontsize=13)
ax1.set_xticks(arange(0, MAX_TC + 1, 5))
ax1.set_ylabel('Phase', labelpad=-10, fontsize=13)
ax1.set_ylim([-pi, pi])
ax1.set_yticks([-pi, pi])
ax1.set_yticklabels(['-π', 'π'])
plt.tight_layout()
plt.show()
