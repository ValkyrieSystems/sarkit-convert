"""
=====================================
Utility functions for SICD converters
=====================================

Common utility functions for use in SICD converters

"""

import itertools

import numpy as np
import numpy.polynomial.polynomial as npp
import sarkit.wgs84

RNIIRS_FIT_PARAMETERS = np.array([3.4761, 0.4357], dtype="float64")


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


def polyfit2d(x, y, z, order1, order2):
    """Fits 2d polynomials to data."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected x and y to be one dimensional")
    if not 0 < z.ndim <= 2:
        raise ValueError("Expected z to be one or two dimensional")
    if not x.shape[0] == y.shape[0] == z.shape[0]:
        raise ValueError("Expected x, y, z to have same leading dimension size")
    vander = npp.polyvander2d(x, y, (order1, order2))
    scales = np.sqrt(np.square(vander).sum(0))
    coefs_flat = (np.linalg.lstsq(vander / scales, z, rcond=-1)[0].T / scales).T
    return coefs_flat.reshape(order1 + 1, order2 + 1)


def polyfit2d_tol(x, y, z, max_order_x, max_order_y, tol, strict_tol=False):
    """Fits 2D polys of minimum order to bring the maximum residual under tol.

    Args
    ----
    x: array-like
        First independent variable values. One dimensional.
    y: array-like
        Second independent variable values. One dimensional.
    z: array-like
        Dependent variable values. Leading dimension must have same size as `x` and `y` .
    max_order_x: int
        The maximum order in `x` to consider
    max_order_y: int
        The maximum order in `y` to consider
    tol: float
        The maximum residual requested.
    strict_tol: bool
        ``True`` if an exception should be raised if `tol` is not met with allowed orders.

        If ``False``, return best fitting polynomial of allowed order.

    Returns
    -------
    poly
        2d polynomials of common orders no greater than `(max_order_x, max_order_y)` .

    Raises
    ------
    `ValueError`
        If `strict_tol` and tolerance is not reached.

    """
    orders = sorted(
        list(itertools.product(range(max_order_x + 1), range(max_order_y + 1))),
        key=lambda x: (x[0] + 1) * (x[1] + 1),
    )
    best = None
    for order_x, order_y in orders:
        poly = polyfit2d(x, y, z, order_x, order_y)
        resid = np.abs(z - np.moveaxis(npp.polyval2d(x, y, poly), 0, -1)).max()
        if resid <= tol:
            return poly
        if best is None or resid < best[1]:
            best = (poly, resid)
    if strict_tol:
        raise ValueError("Max order exceeded before tolerance was reached")
    return best[0]


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


def _get_sigma0_noise(xml_helper):
    """Calculate the absolute noise estimate, in sigma0 power units."""

    if xml_helper.element_tree.find("./{*}Radiometric/{*}SigmaZeroSFPoly") is None:
        raise ValueError(
            "Radiometric.SigmaZeroSFPoly is not populated, so no sigma0 noise estimate can be derived."
        )
    if (
        xml_helper.load("./{*}Radiometric/{*}NoiseLevel/{*}NoiseLevelType")
        != "ABSOLUTE"
    ):
        raise ValueError(
            "Radiometric.NoiseLevel.NoiseLevelType is not `ABSOLUTE` so no noise estimate can be derived."
        )

    noisepoly = xml_helper.load("./{*}Radiometric/{*}NoiseLevel/{*}NoisePoly")
    scp_noise_db = noisepoly[0, 0]
    scp_noise = 10 ** (scp_noise_db / 10)

    # convert to SigmaZero value
    sigma_zero_sf = xml_helper.load("./{*}Radiometric/{*}SigmaZeroSFPoly")
    scp_noise *= sigma_zero_sf[0, 0]

    return scp_noise


def _get_default_signal_estimate(xml_helper):
    """Gets default signal for use in the RNIIRS calculation.

    This will be 1.0 for copolar (or unknown) collections, and 0.25 for cross-pole collections."""

    pol = xml_helper.load("./{*}ImageFormation/{*}TxRcvPolarizationProc")
    if pol is None or ":" not in pol:
        return 1.0

    pols = pol.split(":")

    return 1.0 if pols[0] == pols[1] else 0.25


def _estimate_rniirs(information_density):
    """Calculate an RNIIRS estimate from the information density or Shannon-Hartley channel capacity.

    This mapping has been empirically determined by fitting Shannon-Hartley channel
    capacity to RNIIRS for some sample images.

    To maintain positivity of the estimated rniirs, this transitions to a linear
    model.

    """
    a = RNIIRS_FIT_PARAMETERS
    iim_transition = np.exp(1 - np.log(2) * a[0] / a[1])
    slope = a[1] / (iim_transition * np.log(2))

    if not isinstance(information_density, np.ndarray):
        information_density = np.array(information_density, dtype="float64")
    orig_ndim = information_density.ndim
    if orig_ndim == 0:
        information_density = np.reshape(information_density, (1,))

    out = np.empty(information_density.shape, dtype="float64")
    mask = information_density > iim_transition
    mask_other = ~mask
    if np.any(mask):
        out[mask] = a[0] + a[1] * np.log2(information_density[mask])
    if np.any(mask_other):
        out[mask_other] = slope * information_density[mask_other]

    if orig_ndim == 0:
        return float(out[0])
    return out


def get_rniirs_estimate(xml_helper):
    """This calculates the value(s) for RNIIRS and information density for SICD, according to the RGIQE."""
    scp_noise = _get_sigma0_noise(xml_helper)
    signal = _get_default_signal_estimate(xml_helper)

    u_row = xml_helper.load("./{*}Grid/{*}Row/{*}UVectECF")
    u_col = xml_helper.load("./{*}Grid/{*}Col/{*}UVectECF")
    ipn = np.cross(u_row, u_col)
    u_ipn = ipn / np.linalg.norm(ipn)

    scp_llh = xml_helper.load("./{*}GeoData/{*}SCP/{*}LLH")
    u_gpn = sarkit.wgs84.up(scp_llh)

    bw_sf = np.dot(u_gpn, u_ipn)
    bw_area = abs(
        xml_helper.load("./{*}Grid/{*}Row/{*}ImpRespBW")
        * xml_helper.load("./{*}Grid/{*}Col/{*}ImpRespBW")
        * bw_sf
    )

    inf_density = float(bw_area * np.log2(1 + signal / scp_noise))
    rniirs = float(_estimate_rniirs(inf_density))

    return inf_density, rniirs
