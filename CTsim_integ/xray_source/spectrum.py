"""
Simple tungsten bremsstrahlung spectrum + aluminum filtration.

NOT clinically accurate -- a transparent, simple approximation:
- Kramers' law for the bremsstrahlung shape
- a toy energy-dependent attenuation model for aluminum
- optional tungsten characteristic lines
"""

import numpy as np

Z_TUNGSTEN = 74


def generate_spectrum(kvp, e_min=10.0, e_step=1.0, add_characteristic_peaks=True):
    """
    Simple Kramers-law bremsstrahlung spectrum:

        N(E) ~ Z * (kVp - E) / E     for e_min <= E < kVp

    Returns (energies_keV, intensity) -- unfiltered, unnormalized.
    """
    energies = np.arange(e_min, kvp, e_step)
    intensity = Z_TUNGSTEN * (kvp - energies) / energies
    intensity = np.clip(intensity, 0, None)

    if add_characteristic_peaks and kvp > 69.5:
        # Tungsten K-alpha / K-beta only appear once kVp exceeds the
        # ~69.5 keV K-shell binding energy needed to produce them.
        for peak_E, peak_height in [(59.3, 0.35), (67.2, 0.20)]:
            if peak_E < kvp:
                bump = peak_height * intensity.max() * \
                    np.exp(-0.5 * ((energies - peak_E) / 1.0) ** 2)
                intensity = intensity + bump

    return energies, intensity


def mu_aluminum_per_mm(energy_kev):
    """
    Simple toy linear attenuation coefficient for aluminum (1/mm).
    Captures the two facts we need: much higher attenuation at low
    energy, leveling off at higher energy. NOT from NIST tables --
    good enough for a beam-shaping demo, not for dosimetry.
    """
    energy_kev = np.asarray(energy_kev, dtype=float)
    return 1.5 * (10.0 / energy_kev) ** 3 + 0.015


def apply_filtration(energies, intensity, thickness_mm):
    """Attenuate a spectrum through `thickness_mm` of aluminum."""
    transmission = np.exp(-mu_aluminum_per_mm(energies) * thickness_mm)
    return intensity * transmission


def sample_photon_energies(energies, filtered_intensity, n_samples, rng=None):
    """
    Sample N photon energies from the (already filtered) spectrum.

    np.random.choice / rng.choice with p = normalized spectrum is the
    right tool here: we just need to draw discrete energy bins according
    to their probability. No need for anything more elaborate.
    """
    if rng is None:
        rng = np.random.default_rng()
    probability = filtered_intensity / filtered_intensity.sum()
    return rng.choice(energies, size=n_samples, p=probability)
