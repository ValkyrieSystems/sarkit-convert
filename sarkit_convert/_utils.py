"""
=====================================
Utility functions for SICD converters
=====================================

Common utility functions for use in SICD converters

"""

import numpy as np
import numpy.polynomial.polynomial as npp


def fit_state_vectors(
    fit_time_range, times, positions, velocities=None, accelerations=None, order=5
):
    times = np.asarray(times)
    positions = np.asarray(positions)
    knots_per_state = 1
    if velocities is not None:
        velocities = np.asarray(velocities)
        knots_per_state += 1
    if accelerations is not None:
        accelerations = np.asarray(accelerations)
        knots_per_state += 1

    num_coefs = order + 1
    states_needed = int(np.ceil(num_coefs / knots_per_state))
    if states_needed > times.size:
        raise ValueError("Not enough state vectors")
    start_state = max(np.sum(times < fit_time_range[0]) - 1, 0)
    end_state = min(np.sum(times < fit_time_range[1]) + 1, times.size)
    while end_state - start_state < states_needed:
        start_state = max(start_state - 1, 0)
        end_state = min(end_state + 1, times.size)

    rnc = np.arange(num_coefs)
    used_states = slice(start_state, end_state)
    used_times = times[used_states][:, np.newaxis]
    independent_stack = [used_times**rnc]
    dependent_stack = [positions[used_states, :]]
    if velocities is not None:
        independent_stack.append(rnc * used_times ** (rnc - 1).clip(0))
        dependent_stack.append(velocities[used_states, :])
    if accelerations is not None:
        independent_stack.append(rnc * (rnc - 1) * used_times ** (rnc - 2).clip(0))
        dependent_stack.append(accelerations[used_states, :])

    dependent = np.stack(dependent_stack, axis=-2)
    independent = np.stack(independent_stack, axis=-2)
    return np.linalg.lstsq(
        independent.reshape(-1, independent.shape[-1]),
        dependent.reshape(-1, dependent.shape[-1]),
        rcond=-1,
    )[0]


def polyfit2d(x, y, z, deg1, deg2):
    """Fits 2d polynomials to data.

    Example
    -------
    >>> x, y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-2, 2, 53), indexing="ij")
    >>> poly = np.ones((3, 4))
    >>> z = npp.polyval2d(x, y, poly)
    >>> np.allclose(polyfit2d(x.flatten(), y.flatten(), z.reshape(2, -1).T, 2, 3), poly)
    True

    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected x and y to be one dimensional")
    if not 0 < z.ndim <= 2:
        raise ValueError("Expected z to be one or two dimensional")
    if not x.shape[0] == y.shape[0] == z.shape[0]:
        raise ValueError("Expected x, y, z to have same leading dimension size")
    vander = npp.polyvander2d(x, y, (deg1, deg2))
    scales = np.sqrt(np.square(vander).sum(0))
    coefs_flat = (np.linalg.lstsq(vander / scales, z, rcond=-1)[0].T / scales).T
    return coefs_flat.reshape(deg1 + 1, deg2 + 1)


def broadening_from_amp(amp_vals, threshold_db=None):
    """Compute the broadening factor from amplitudes

    Parameters
    ----------
    amp_vals: array-like
        window amplitudes
    threshold_db: float, optional
        threshold to use to compute broadening (Default: 10*log10(0.5))

    Returns
    -------
    float

    """
    if threshold_db is None:
        threshold = np.sqrt(0.5)
    else:
        threshold = 10 ** (threshold_db / 20)
    amp_vals = np.asarray(amp_vals)
    fft_size = 2 ** int(np.ceil(np.log2(amp_vals.size * 10000)))
    impulse_response = np.abs(np.fft.fft(amp_vals, fft_size))
    impulse_response /= impulse_response.max()
    width = (impulse_response[: fft_size // 2] < threshold).argmax() + (
        impulse_response[-1 : fft_size // 2 : -1] > threshold
    ).argmin()

    return width / fft_size * amp_vals.size
