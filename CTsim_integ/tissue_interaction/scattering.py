"""
scattering_corrected.py
-----------------------
Energy-resolved, detector-space scatter approximation for CT simulation.

Input shape:
    (detector_rows, detector_channels, energy_bins)

Output:
    Primary fluence
    Scatter fluence
    Total fluence
    Scatter kernel
    Per-energy SPR statistics

IMPORTANT:
This is a phenomenological scatter-kernel approximation, not a
Monte Carlo simulation or a full physical Compton-scattering model.
The SPR parameter must be calibrated for the simulated geometry.
"""

import numpy as np
from scipy.signal import fftconvolve


def _make_scatter_kernel(
    kernel_size=21,
    alpha=0.7,
    beta=0.08,
    gamma=0.3,
    delta=0.35,
):
    """
    Construct a normalized radial scatter kernel:

        K(r) = alpha * exp(-beta * r^2)
             + gamma * exp(-delta * r)
    """

    if not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be an integer.")

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")

    parameters = (alpha, beta, gamma, delta)

    if not all(np.isfinite(v) and v >= 0 for v in parameters):
        raise ValueError(
            "alpha, beta, gamma and delta must be finite and non-negative."
        )

    radius = kernel_size // 2

    yy, xx = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    r = np.sqrt(xx**2 + yy**2)

    kernel = (
        alpha * np.exp(-beta * r**2)
        + gamma * np.exp(-delta * r)
    )

    kernel_sum = float(kernel.sum())

    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError("Scatter kernel has invalid total weight.")

    return kernel / kernel_sum


def _convolve_reflect(image, kernel):
    """
    Apply 2D convolution with reflection padding.

    Reflection padding reduces artificial signal loss at the detector edges.
    """

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="reflect"
    )

    blurred_padded = fftconvolve(
        padded,
        kernel,
        mode="same"
    )

    blurred = blurred_padded[
        pad_y:pad_y + image.shape[0],
        pad_x:pad_x + image.shape[1]
    ]

    return blurred


def add_scatter(
    primary_photon_fluence,
    spr=1.2,
    kernel_size=21,
    alpha=0.7,
    beta=0.08,
    gamma=0.3,
    delta=0.35,
    normalize_spr=True,
):
    """
    Estimate scatter and combine it with primary photon fluence.

    Parameters
    ----------
    primary_photon_fluence : ndarray
        Shape (detector_rows, detector_channels, energy_bins).
        Must contain energy-resolved primary fluence.

    spr : float
        Target scatter-to-primary ratio.
        1.2 is a configurable starting value, not a universal CT constant.

    kernel_size : int
        Odd spatial kernel size in detector pixels.

    alpha, beta, gamma, delta : float
        Parameters controlling the radial kernel shape.

    normalize_spr : bool
        If True, normalize the scatter sum in each energy bin so that:

            sum(scatter_E) = spr * sum(primary_E)

    Returns
    -------
    dict
        Primary, scatter, total, kernel and per-energy statistics.
    """

    primary = np.asarray(
        primary_photon_fluence,
        dtype=np.float64
    )

    # Validate dimensions.
    if primary.ndim != 3:
        raise ValueError(
            "Expected shape (detector_rows, detector_channels, energy_bins); "
            f"received {primary.shape}."
        )

    if any(size == 0 for size in primary.shape):
        raise ValueError("Input dimensions must be non-zero.")

    # Validate fluence.
    if not np.all(np.isfinite(primary)):
        raise ValueError("Primary fluence contains NaN or infinite values.")

    if np.any(primary < 0):
        raise ValueError("Primary fluence cannot contain negative values.")

    # Validate SPR.
    spr = float(spr)

    if not np.isfinite(spr) or spr < 0:
        raise ValueError("SPR must be finite and non-negative.")

    # Construct scatter kernel.
    kernel = _make_scatter_kernel(
        kernel_size=kernel_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )

    rows, channels, n_energies = primary.shape

    scatter = np.zeros_like(primary)

    # Process each energy bin independently.
    for energy_index in range(n_energies):

        primary_image = primary[:, :, energy_index]

        primary_sum = float(primary_image.sum())

        if primary_sum == 0.0 or spr == 0.0:
            continue

        # Spatial redistribution of primary fluence.
        blurred = _convolve_reflect(
            primary_image,
            kernel
        )

        # Suppress tiny negative numerical errors from FFT convolution.
        blurred = np.maximum(blurred, 0.0)

        if normalize_spr:

            blurred_sum = float(blurred.sum())

            if blurred_sum > 0.0:
                scatter[:, :, energy_index] = (
                    blurred * (spr * primary_sum / blurred_sum)
                )

        else:

            scatter[:, :, energy_index] = spr * blurred

    # Add scatter to primary exactly once.
    total = primary + scatter

    # Calculate energy-resolved statistics.
    primary_by_energy = primary.sum(axis=(0, 1))
    scatter_by_energy = scatter.sum(axis=(0, 1))
    total_by_energy = total.sum(axis=(0, 1))

    actual_spr_by_energy = np.divide(
        scatter_by_energy,
        primary_by_energy,
        out=np.zeros_like(scatter_by_energy),
        where=primary_by_energy > 0
    )

    return {
        "primary_photon_fluence": primary,
        "scatter_photon_fluence": scatter,
        "total_photon_fluence": total,
        "scatter_kernel": kernel,
        "primary_total_by_energy": primary_by_energy,
        "scatter_total_by_energy": scatter_by_energy,
        "total_total_by_energy": total_by_energy,
        "actual_spr_by_energy": actual_spr_by_energy,
    }


if __name__ == "__main__":

    # Standalone test using synthetic energy-resolved fluence.
    # This does NOT load or modify your CT simulation outputs.

    rng = np.random.default_rng(42)

    test_primary = rng.uniform(
        low=1e5,
        high=1e6,
        size=(64, 888, 10)
    )

    result = add_scatter(
        test_primary,
        spr=1.2,
        kernel_size=21,
        normalize_spr=True
    )

    primary = result["primary_photon_fluence"]
    scatter = result["scatter_photon_fluence"]
    total = result["total_photon_fluence"]

    overall_spr = (
        scatter.sum() / primary.sum()
        if primary.sum() > 0
        else 0.0
    )

    print("\n========== SCATTER MODEL TEST ==========")
    print("Primary shape:", primary.shape)
    print("Scatter shape:", scatter.shape)
    print("Total shape:", total.shape)

    print(f"Primary fluence: {primary.sum():.6e}")
    print(f"Scatter fluence: {scatter.sum():.6e}")
    print(f"Total fluence:   {total.sum():.6e}")
    print(f"Overall SPR:     {overall_spr:.6f}")

    print(
        "Addition check:",
        np.allclose(total, primary + scatter)
    )

    print(
        "All values finite:",
        np.all(np.isfinite(total))
    )

    print(
        "All values non-negative:",
        np.all(total >= 0)
    )

    print("\nActual SPR by energy bin:")
    print(result["actual_spr_by_energy"])

    print("========================================")