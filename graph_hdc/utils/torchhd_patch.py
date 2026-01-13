"""
TorchHD HRR Patches for correct multibind/multibundle operations.

The default HRRTensor.multibind implementation has a bug where it casts
complex FFT results to real dtype prematurely, corrupting the circular
convolution. This patch fixes that issue.
"""

import torch
from torchhd import HRRTensor

_orig_multibind = HRRTensor.multibind
_orig_multibundle = HRRTensor.multibundle


def hrr_multibind_corrected(self) -> "HRRTensor":
    """
    Corrected multibind that properly handles complex FFT multiplication.

    The original implementation incorrectly casts to real dtype during
    the frequency-domain product, losing imaginary components needed for
    correct circular convolution.
    """
    # Compute FFT of all vectors (complex result)
    spectra = torch.fft.fft(self, dim=-1)
    # Multiply spectra across vectors (no dtype cast to float!)
    prod_spectra = torch.prod(spectra, dim=-2)
    # Inverse FFT to get convolution result
    result = torch.fft.ifft(prod_spectra, dim=-1)
    # Take the real part
    return torch.real(result).as_subclass(HRRTensor)


def patched_multibind(self):
    """Multibind with identity short-circuit for single-slot inputs."""
    if self.size(-2) == 1:
        return self.squeeze(-2)
    return hrr_multibind_corrected(self)


def patched_multibundle(self):
    """Multibundle with identity short-circuit for single-slot inputs."""
    if self.size(-2) == 1:
        return self.squeeze(-2)
    return _orig_multibundle(self)


def apply_patches():
    """Apply the HRR patches to torchhd."""
    HRRTensor.multibind = patched_multibind
    HRRTensor.multibundle = patched_multibundle


# Apply patches on import
apply_patches()
