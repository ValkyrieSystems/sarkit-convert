"""
=====================
Iceye Complex to SICD
=====================

Convert a complex image from the Iceye HD5 SLC into SICD.

Note: In the development of this converter "Iceye Product Metadata" description (v2.1, v2.2, v2.4, v2.5) was considered.

"""

import argparse
import copy
import pathlib

import dateutil.parser
import h5py
import lxml.builder
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import sarkit.wgs84
from sarkit import _constants
from sarkit.verification import SicdConsistency

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}

PIXEL_TYPE_MAP = {
    "float32": "RE32F_IM32F",
    "int16": "RE16I_IM16I",
}


def _extract_attributes(h5_obj):
    """Recursively extract all dataset names and values into a dictionary, skipping specified keys and decoding byte strings."""
    result = {}

    for key in h5_obj:  # Iterate over keys in the HDF5 group
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset):
            value = item[...]
            if isinstance(value, bytes):
                value = value.decode("utf-8")  # Decode byte string
            elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
                value = value.astype(str).tolist()  # Handle ndarrays with type string
            elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
                value = value.item().decode("utf-8")  # Handle ndarrays with type object
            elif isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()  # Handle single value arrays

            result[key] = value
        elif isinstance(item, h5py.Group):  # If it is a group, recurse into it
            if np.array_equal(item.attrs.get("type"), [b"hickle"]):
                result[key] = None
            else:
                result[key] = _extract_attributes(item)

    return result


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


def compute_apc_poly(h5_attrs, start_time, stop_time):
    """Creates an Aperture Phase Center (APC) poly that orbits the Earth above the equator.

    Polynomial generates 3D coords in ECF as a function of time from start of collect.

    Parameters
    ----------
    h5_attrs: dict
        The collection metadata
    start_time: float
        The start time to fit.
    stop_time: float
        The end time to fit.

    Returns
    -------
    `numpy.ndarray`, shape=(6, 3)
        APC poly
    """
    times_str = np.array(h5_attrs["state_vector_time_utc"]).flatten()
    state_times = np.asarray(
        [dateutil.parser.parse(entry) for entry in times_str], dtype=np.datetime64
    )
    times = (state_times - np.datetime64(start_time)) / np.timedelta64(1, "s")
    positions = np.zeros((times.size, 3), dtype="float64")
    velocities = np.zeros((times.size, 3), dtype="float64")

    positions[:, :] = np.stack(
        (h5_attrs["posX"], h5_attrs["posY"], h5_attrs["posZ"]), axis=1
    )
    velocities[:, :] = np.stack(
        (h5_attrs["velX"], h5_attrs["velY"], h5_attrs["velZ"]), axis=1
    )

    apc_poly = fit_state_vectors(
        (0, (stop_time - start_time).total_seconds()),
        times,
        positions,
        velocities,
        order=5,
    )

    return apc_poly


def polyfit2d(x, y, z, deg1, deg2):
    """Fits 2d polynomials to data.

    Example
    -------
    >>> x, y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-2, 2, 53), indexing="ij")
    >>> poly = np.ones((3, 4, 2))
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


def _update_radiometric_node(sicd_xmltree):
    """Use existing metadata to populate the Radiometric XML node."""
    xmlhelp = sksicd.XmlHelper(copy.deepcopy(sicd_xmltree))

    def get_slant_plane_area(xmlhelp):
        row_imp_resp_bw = xmlhelp.load("./{*}Grid/{*}Row/{*}ImpRespBW")
        col_imp_resp_bw = xmlhelp.load("./{*}Grid/{*}Col/{*}ImpRespBW")
        range_weight_f = azimuth_weight_f = 1.0
        row_wgt_funct = xmlhelp.load("./{*}Grid/{*}Row/{*}WgtFunct")
        if row_wgt_funct is not None:
            var = np.var(row_wgt_funct)
            mean = np.mean(row_wgt_funct)
            range_weight_f += var / (mean * mean)
        col_wgt_funct = xmlhelp.load("./{*}Grid/{*}Col/{*}WgtFunct")
        if col_wgt_funct is not None:
            var = np.var(col_wgt_funct)
            mean = np.mean(col_wgt_funct)
            azimuth_weight_f += var / (mean * mean)
        return (range_weight_f * azimuth_weight_f) / (row_imp_resp_bw * col_imp_resp_bw)

    sp_area = get_slant_plane_area(xmlhelp)
    radiometric_node = xmlhelp.element_tree.find("./{*}Radiometric")
    scpcoa_slope_ang = xmlhelp.load("./{*}SCPCOA/{*}SlopeAng")
    scpcoa_graze_ang = xmlhelp.load("./{*}SCPCOA/{*}GrazeAng")
    if radiometric_node.find("{*}BetaZeroSFPoly") is None:
        if radiometric_node.find("{*}RCSSFPoly") is not None:
            beta_zero_sf_poly_coefs = (
                xmlhelp.load_elem(radiometric_node.find("{*}RCSSFPoly")) / sp_area
            )
        elif radiometric_node.find("{*}SigmaZeroSFPoly") is not None:
            beta_zero_sf_poly_coefs = xmlhelp.load_elem(
                radiometric_node.find("{*}SigmaZeroSFPoly")
            ) / np.cos(np.deg2rad(scpcoa_slope_ang))
        elif radiometric_node.find("{*}GammaZeroSFPoly") is not None:
            beta_zero_sf_poly_coefs = xmlhelp.load_elem(
                radiometric_node.find("{*}GammaZeroSFPoly")
            ) * (
                np.sin(np.deg2rad(scpcoa_graze_ang))
                / np.cos(np.deg2rad(scpcoa_slope_ang))
            )
    else:
        beta_zero_sf_poly_coefs = xmlhelp.load_elem(
            radiometric_node.find("{*}BetaZeroSFPoly")
        )

    if beta_zero_sf_poly_coefs is not None:
        # In other words, none of the SF polynomials are populated.
        if radiometric_node.find("{*}RCSSFPoly") is None:
            rcs_sf_poly_coefs = beta_zero_sf_poly_coefs * sp_area
        if radiometric_node.find("{*}SigmaZeroSFPoly") is None:
            sigma_zero_sf_poly_coefs = beta_zero_sf_poly_coefs * np.cos(
                np.deg2rad(scpcoa_slope_ang)
            )
        if radiometric_node.find("{*}GammaZeroSFPoly") is None:
            gamma_zero_sf_poly_coefs = beta_zero_sf_poly_coefs * (
                np.cos(np.deg2rad(scpcoa_slope_ang))
                / np.sin(np.deg2rad(scpcoa_graze_ang))
            )

    sicd = lxml.builder.ElementMaker(
        namespace=NSMAP["sicd"], nsmap={None: NSMAP["sicd"]}
    )
    new_radiometric_node = sicd.Radiometric(
        sicd.RCSSFPoly(),
        sicd.SigmaZeroSFPoly(),
        sicd.BetaZeroSFPoly(),
        sicd.GammaZeroSFPoly(),
    )
    sksicd.Poly2dType().set_elem(
        new_radiometric_node.find("./{*}RCSSFPoly"), rcs_sf_poly_coefs
    )
    sksicd.Poly2dType().set_elem(
        new_radiometric_node.find("./{*}SigmaZeroSFPoly"), sigma_zero_sf_poly_coefs
    )
    sksicd.Poly2dType().set_elem(
        new_radiometric_node.find("./{*}BetaZeroSFPoly"), beta_zero_sf_poly_coefs
    )
    sksicd.Poly2dType().set_elem(
        new_radiometric_node.find("./{*}GammaZeroSFPoly"), gamma_zero_sf_poly_coefs
    )

    return new_radiometric_node


def _get_x_y_coords(num_row_col, spacing_row_col, scp_pixel, start_row_col):
    """Create the X, Y coordinates of the full image"""
    full_img_verticies = np.array(
        [
            [0, 0],
            [0, num_row_col[1] - 1],
            [num_row_col[0] - 1, num_row_col[1] - 1],
            [num_row_col[0] - 1, 0],
        ],
    )

    x_coords = spacing_row_col[0] * (
        full_img_verticies[:, 0] - (scp_pixel[0] - start_row_col[0])
    )
    y_coords = spacing_row_col[1] * (
        full_img_verticies[:, 1] - (scp_pixel[1] - start_row_col[1])
    )

    return x_coords, y_coords


def _calc_deltaks(x_coords, y_coords, deltak_coa_poly, imp_resp_bw, spacing):
    """Calculate the minimum and maximum DeltaK values"""
    deltaks = npp.polyval2d(x_coords, y_coords, deltak_coa_poly)
    min_deltak = np.amin(deltaks) - 0.5 * imp_resp_bw
    max_deltak = np.amax(deltaks) + 0.5 * imp_resp_bw

    if (min_deltak < -0.5 / abs(spacing)) or (max_deltak > 0.5 / abs(spacing)):
        min_deltak = -0.5 / abs(spacing)
        max_deltak = -min_deltak

    return min_deltak, max_deltak


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


def hdf5_to_sicd(h5_filename, sicd_filename, classification, ostaid="Unknown"):
    """Converts Iceye native SLC h5 files to NGA standard SICD files.

    Parameters
    ----------
    h5_filename: str
        path of the input HDF5 file
    sicd_filename: str
        path of the output SICD file.
    classification: str
        content of the /SICD/CollectionInfo/Classification node in the SICD XML.
    ostaid: str, optional
        content of the originating station ID (OSTAID) field of the NITF header.

    """
    with h5py.File(h5_filename, "r") as h5file:
        h5_attrs = _extract_attributes(h5file)

    # Timeline
    collect_start = dateutil.parser.parse(h5_attrs["acquisition_start_utc"])
    collect_stop = dateutil.parser.parse(h5_attrs["acquisition_end_utc"])
    collect_duration = (collect_stop - collect_start).total_seconds()
    acq_prf = h5_attrs["acquisition_prf"]
    num_pulses = int(np.round(collect_duration * acq_prf))
    t_start = 0
    t_end = collect_duration
    ipp_start = 0
    ipp_end = int(num_pulses - 1)
    ipp_poly = [0, acq_prf]
    look = {"left": 1, "right": -1}[h5_attrs["look_side"].lower()]

    # Collection Info
    collector_name = h5_attrs["satellite_name"]
    core_name = h5_attrs["product_name"]
    collect_type = "MONOSTATIC"
    mode_id = h5_attrs["product_type"]
    mode_type = h5_attrs["acquisition_mode"].upper()
    if not mode_type:
        mode_type = "DYNAMIC STRIPMAP"

    # Creation Info
    creation_application = f"ICEYE_P_{h5_attrs['processor_version']}"
    creation_date_time = dateutil.parser.parse(h5_attrs["processing_time"])

    # Image Data
    samp_prec = h5_attrs["sample_precision"]
    pixel_type = PIXEL_TYPE_MAP[samp_prec]
    num_rows = int(h5_attrs["number_of_range_samples"])
    num_cols = int(h5_attrs["number_of_azimuth_samples"])
    first_row = 0
    first_col = 0
    scp_pixel = (num_rows // 2, num_cols // 2)

    # # Geo Data
    coord_center = h5_attrs["coord_center"]
    avg_scene_height = float(h5_attrs["avg_scene_height"])
    init_scp_llh = [coord_center[2], coord_center[3], avg_scene_height]

    # Position
    apc_poly = compute_apc_poly(h5_attrs, collect_start, collect_stop)

    # Radar Collection
    center_frequency = h5_attrs["carrier_frequency"]
    tx_rf_bw = h5_attrs["chirp_bandwidth"]
    tx_freq_min = center_frequency - 0.5 * tx_rf_bw
    tx_freq_max = center_frequency + 0.5 * tx_rf_bw
    tx_pulse_length = h5_attrs["chirp_duration"]
    rcv_demod_type = "CHIRP"
    adc_sample_rate = h5_attrs["range_sampling_rate"]
    tx_fm_rate = tx_rf_bw / tx_pulse_length
    tx_polarization = h5_attrs["polarization"][0]
    rcv_polarization = h5_attrs["polarization"][1]
    tx_rcv_polarization = f"{tx_polarization}:{rcv_polarization}"

    row_bw = 2 * tx_rf_bw / _constants.speed_of_light

    # Image Formation
    tx_rcv_polarization_proc = tx_rcv_polarization
    image_form_algo = "RMA"
    t_start_proc = 0
    t_end_proc = collect_duration
    tx_freq_proc = (tx_freq_min, tx_freq_max)
    st_beam_comp = "NO"
    image_beam_comp = "SV"
    az_autofocus = "NO"
    rg_autofocus = "NO"

    def calculate_drate_polys():
        r_ca_coefs = np.array([r_ca_scp, 1], dtype="float64")
        doppler_rate_coefs = h5_attrs["doppler_rate_coeffs"]
        # Prior to ICEYE 1.14 processor, absolute value of Doppler rate was
        # provided, not true Doppler rate. Doppler rate should always be negative
        if doppler_rate_coefs[0] > 0:
            doppler_rate_coefs *= -1
        dop_rate_poly = doppler_rate_coefs

        def shift(coefs, t_0: float, alpha: float = 1):
            # prepare array workspace
            out = np.copy(coefs)
            if t_0 != 0 and out.size > 1:
                siz = out.size
                for i in range(siz):
                    index = siz - i - 1
                    if i > 0:
                        out[index : siz - 1] -= t_0 * out[index + 1 : siz]

            if alpha != 1 and out.size > 1:
                out *= np.power(alpha, np.arange(out.size))

            return out

        drate_ca_poly_coefs = shift(
            dop_rate_poly,
            t_0=zd_ref_time - rg_time_scp,
            alpha=2 / _constants.speed_of_light,
        )

        drsf_poly_coefs = (
            -npp.polymul(drate_ca_poly_coefs, r_ca_coefs)
            * _constants.speed_of_light
            / (2 * center_frequency * vm_ca_sq)
        )

        return drate_ca_poly_coefs, drsf_poly_coefs

    def calculate_doppler_polys():
        # define and fit the time coa array
        if mode_type == "SPOTLIGHT":
            coa_time = collect_duration / 2
            alpha = 2.0 / _constants.speed_of_light
            pos = npp.polyval(coa_time, apc_poly)
            vel = npp.polyval(coa_time, npp.polyder(apc_poly))
            speed = np.linalg.norm(vel)
            vel_hat = vel / speed
            scp = sarkit.wgs84.geodetic_to_cartesian(
                [coord_center[2], coord_center[3], avg_scene_height]
            )
            los = scp - pos

            time_coa_poly_coefs = np.array(
                [
                    [
                        coa_time,
                    ],
                ]
            )
            dop_centroid_poly_coefs = np.zeros((2, 2), dtype=np.float64)
            dop_centroid_poly_coefs[0, 1] = (
                -look * center_frequency * alpha * speed / r_ca_scp
            )
            dop_centroid_poly_coefs[1, 1] = (
                look * center_frequency * alpha * speed / (r_ca_scp**2)
            )
            dop_centroid_poly_coefs[:, 0] = -look * (
                dop_centroid_poly_coefs[:, 1] * np.dot(los, vel_hat)
            )
        else:
            # extract doppler centroid coefficients
            dc_estimate_coefs = h5_attrs["dc_estimate_coeffs"]
            dc_time_str = h5_attrs["dc_estimate_time_utc"]
            dc_zd_times = np.zeros((len(dc_time_str),), dtype="float64")
            for i, entry in enumerate(dc_time_str):
                dc_zd_times[i] = (
                    dateutil.parser.parse(entry[0]) - collect_start
                ).total_seconds()
            # create a sampled doppler centroid
            samples = 51
            # create doppler time samples
            diff_time_rg = (
                first_pixel_time
                - zd_ref_time
                + np.linspace(0, num_rows / adc_sample_rate, samples)
            )
            # doppler centroid samples definition
            dc_sample_array = np.zeros((samples, dc_zd_times.size), dtype="float64")
            for i, coefs in enumerate(dc_estimate_coefs):
                dc_sample_array[:, i] = npp.polyval(diff_time_rg, coefs)
            # create arrays for range/azimuth from scp in meters
            azimuth_scp_m, range_scp_m = np.meshgrid(
                col_ss * (dc_zd_times - zd_time_scp) / ss_zd_s,
                (diff_time_rg + zd_ref_time - rg_time_scp)
                * _constants.speed_of_light
                / 2,
            )

            x_order = min(3, range_scp_m.shape[0] - 1)
            y_order = min(3, range_scp_m.shape[1] - 1)

            # fit the doppler centroid sample array
            dop_centroid_poly_coefs = polyfit2d(
                range_scp_m.flatten(),
                azimuth_scp_m.flatten(),
                dc_sample_array.flatten(),
                x_order,
                y_order,
            )
            doppler_rate_sampled = npp.polyval(azimuth_scp_m, drca_poly_coefs)
            time_coa = dc_zd_times + dc_sample_array / doppler_rate_sampled
            time_coa_poly_coefs = polyfit2d(
                range_scp_m.flatten(),
                azimuth_scp_m.flatten(),
                time_coa.flatten(),
                x_order,
                y_order,
            )

        return dop_centroid_poly_coefs, time_coa_poly_coefs

    ss_zd_s = float(h5_attrs["azimuth_time_interval"])
    if look == 1:
        ss_zd_s *= -1
        zero_doppler_left = dateutil.parser.parse(h5_attrs["zerodoppler_end_utc"])
    else:
        zero_doppler_left = dateutil.parser.parse(h5_attrs["zerodoppler_start_utc"])
    dop_bw = h5_attrs["total_processed_bandwidth_azimuth"]
    zd_time_scp = (zero_doppler_left - collect_start).total_seconds() + scp_pixel[
        1
    ] * ss_zd_s
    first_pixel_time = float(h5_attrs["first_pixel_time"])
    zd_ref_time = first_pixel_time + num_rows / (2 * adc_sample_rate)
    vel_scp = npp.polyval(zd_time_scp, npp.polyder(apc_poly))
    vm_ca_sq = np.sum(vel_scp * vel_scp)
    rg_time_scp = first_pixel_time + scp_pixel[0] / adc_sample_rate
    r_ca_scp = rg_time_scp * _constants.speed_of_light / 2
    drca_poly_coefs, drsf_poly_coefs = calculate_drate_polys()

    # calculate some doppler dependent grid parameters
    col_ss = float(np.sqrt(vm_ca_sq) * abs(ss_zd_s) * drsf_poly_coefs[0])
    col_bw = dop_bw * abs(ss_zd_s) / col_ss
    time_ca_poly_coefs = [zd_time_scp, ss_zd_s / col_ss]

    # RMA
    dop_centroid_poly_coefs, time_coa_poly_coefs = calculate_doppler_polys()
    if mode_type == "SPOTLIGHT":
        dop_centroid_poly = np.array([[0]])
        dop_centroid_coa = "false"
    else:
        dop_centroid_poly = dop_centroid_poly_coefs
        dop_centroid_coa = "true"
    first_pixel_time = h5_attrs["first_pixel_time"]
    rg_time_scp = first_pixel_time + adc_sample_rate
    freq_zero = center_frequency
    dr_sf_poly = drsf_poly_coefs[:, np.newaxis]
    time_ca_poly = time_ca_poly_coefs
    rma_algo_type = "OMEGA_K"
    image_type = "INCA"

    # Grid
    image_plane = "SLANT"
    grid_type = "RGZERO"
    row_ss = _constants.speed_of_light / (2 * adc_sample_rate)
    row_imp_res_bw = row_bw
    row_sgn = -1
    row_kctr = str(center_frequency / (_constants.speed_of_light / 2))
    row_deltak_coa_poly = np.array([[0]])

    col_ss = col_ss
    col_imp_res_bw = col_bw
    col_sgn = -1
    col_kctr = 0
    time_coa_poly = time_coa_poly_coefs
    col_deltak_coa_poly = dop_centroid_poly_coefs * ss_zd_s / col_ss

    row_win = h5_attrs["window_function_range"]
    col_win = h5_attrs["window_function_azimuth"]
    if row_win == "NONE":
        row_win = "UNIFORM"
    if col_win == "NONE":
        col_win = "UNIFORM"
    row_brodening_factor = broadening_from_amp(np.ones(256))
    col_brodening_factor = broadening_from_amp(np.ones(256))
    row_imp_res_wid = row_brodening_factor / row_imp_res_bw
    col_imp_res_wid = col_brodening_factor / col_imp_res_bw

    x_coords, y_coords = _get_x_y_coords(
        [num_rows, num_cols], [row_ss, col_ss], scp_pixel, [first_row, first_col]
    )

    row_delta_k1, row_delta_k2 = _calc_deltaks(
        x_coords, y_coords, row_deltak_coa_poly, row_imp_res_bw, row_ss
    )
    col_delta_k1, col_delta_k2 = _calc_deltaks(
        x_coords, y_coords, col_deltak_coa_poly, col_imp_res_bw, col_ss
    )

    # Adjust SCP
    scp_drsf = dr_sf_poly[0, 0]
    scp_tca = time_ca_poly[0]
    scp_tcoa = time_coa_poly[0, 0]
    scp_delta_t_coa = scp_tcoa - scp_tca
    scp_varp_ca_mag = npl.norm(npp.polyval(scp_tca, npp.polyder(apc_poly)))
    scp_rcoa = np.sqrt(r_ca_scp**2 + scp_drsf * scp_varp_ca_mag**2 * scp_delta_t_coa**2)
    scp_rratecoa = scp_drsf / scp_rcoa * scp_varp_ca_mag**2 * scp_delta_t_coa
    scp_set = sksicd.projection.ProjectionSetsMono(
        t_COA=np.array([scp_tcoa]),
        ARP_COA=np.array([npp.polyval(scp_tcoa, apc_poly)]),
        VARP_COA=np.array([npp.polyval(scp_tcoa, npp.polyder(apc_poly))]),
        R_COA=np.array([scp_rcoa]),
        Rdot_COA=np.array([scp_rratecoa]),
    )
    scp_ecf = sksicd.projection.r_rdot_to_ground_plane_mono(
        look,
        scp_set,
        sarkit.wgs84.geodetic_to_cartesian(init_scp_llh),
        sarkit.wgs84.up(init_scp_llh),
    )[0]
    scp_llh = sarkit.wgs84.cartesian_to_geodetic(scp_ecf)

    # Calc unit vectors
    scp_ca_pos = npp.polyval(scp_tca, apc_poly)
    scp_ca_vel = npp.polyval(scp_tca, npp.polyder(apc_poly))
    los = scp_ecf - scp_ca_pos
    row_uvect_ecf = los / npl.norm(los)
    left = np.cross(scp_ca_pos, scp_ca_vel)
    look = np.sign(np.dot(left, row_uvect_ecf))
    spz = -look * np.cross(row_uvect_ecf, scp_ca_vel)
    uspz = spz / npl.norm(spz)
    col_uvect_ecf = np.cross(uspz, row_uvect_ecf)

    # Radiometric
    beta_zero_sf_poly = [
        [
            float(h5_attrs["calibration_factor"]),
        ],
    ]

    # Build XML
    sicd = lxml.builder.ElementMaker(
        namespace=NSMAP["sicd"], nsmap={None: NSMAP["sicd"]}
    )
    collection_info = sicd.CollectionInfo(
        sicd.CollectorName(collector_name),
        sicd.CoreName(core_name),
        sicd.CollectType(collect_type),
        sicd.RadarMode(sicd.ModeType(mode_type), sicd.ModeID(mode_id)),
        sicd.Classification(classification),
    )
    image_creation = sicd.ImageCreation(
        sicd.Application(creation_application),
        sicd.DateTime(creation_date_time.isoformat() + "Z"),
        sicd.Site(ostaid),
    )
    image_data = sicd.ImageData(
        sicd.PixelType(pixel_type),
        sicd.NumRows(str(num_rows)),
        sicd.NumCols(str(num_cols)),
        sicd.FirstRow(str(first_row)),
        sicd.FirstCol(str(first_col)),
        sicd.FullImage(sicd.NumRows(str(num_rows)), sicd.NumCols(str(num_cols))),
        sicd.SCPPixel(sicd.Row(str(scp_pixel[0])), sicd.Col(str(scp_pixel[1]))),
    )

    def make_xyz(arr):
        return [sicd.X(str(arr[0])), sicd.Y(str(arr[1])), sicd.Z(str(arr[2]))]

    def make_llh(arr):
        return [sicd.Lat(str(arr[0])), sicd.Lon(str(arr[1])), sicd.HAE(str(arr[2]))]

    def make_ll(arr):
        return [sicd.Lat(str(arr[0])), sicd.Lon(str(arr[1]))]

    # Add GeoData with placeholder corners
    geo_data = sicd.GeoData(
        sicd.EarthModel("WGS_84"),
        sicd.SCP(sicd.ECF(*make_xyz(scp_ecf)), sicd.LLH(*make_llh(scp_llh))),
        sicd.ImageCorners(
            sicd.ICP({"index": "1:FRFC"}, *make_ll([0, 0])),
            sicd.ICP({"index": "2:FRLC"}, *make_ll([0, 0])),
            sicd.ICP({"index": "3:LRLC"}, *make_ll([0, 0])),
            sicd.ICP({"index": "4:LRFC"}, *make_ll([0, 0])),
        ),
    )

    grid = sicd.Grid(
        sicd.ImagePlane(image_plane),
        sicd.Type(grid_type),
        sicd.TimeCOAPoly(),
        sicd.Row(
            sicd.UVectECF(*make_xyz(row_uvect_ecf)),
            sicd.SS(str(row_ss)),
            sicd.ImpRespWid(str(row_imp_res_wid)),
            sicd.Sgn(str(row_sgn)),
            sicd.ImpRespBW(str(row_imp_res_bw)),
            sicd.KCtr(str(row_kctr)),
            sicd.DeltaK1(str(row_delta_k1)),
            sicd.DeltaK2(str(row_delta_k2)),
            sicd.DeltaKCOAPoly(),
            sicd.WgtType(
                sicd.WindowName(
                    str(row_win),
                )
            ),
        ),
        sicd.Col(
            sicd.UVectECF(*make_xyz(col_uvect_ecf)),
            sicd.SS(str(col_ss)),
            sicd.ImpRespWid(str(col_imp_res_wid)),
            sicd.Sgn(str(col_sgn)),
            sicd.ImpRespBW(str(col_imp_res_bw)),
            sicd.KCtr(str(col_kctr)),
            sicd.DeltaK1(str(col_delta_k1)),
            sicd.DeltaK2(str(col_delta_k2)),
            sicd.DeltaKCOAPoly(),
            sicd.WgtType(
                sicd.WindowName(
                    str(col_win),
                )
            ),
        ),
    )
    sksicd.Poly2dType().set_elem(grid.find("./{*}TimeCOAPoly"), time_coa_poly)
    sksicd.Poly2dType().set_elem(
        grid.find("./{*}Row/{*}DeltaKCOAPoly"), row_deltak_coa_poly
    )
    sksicd.Poly2dType().set_elem(
        grid.find("./{*}Col/{*}DeltaKCOAPoly"), col_deltak_coa_poly
    )

    timeline = sicd.Timeline(
        sicd.CollectStart(collect_start.isoformat() + "Z"),
        sicd.CollectDuration(str(collect_duration)),
        sicd.IPP(
            {"size": "1"},
            sicd.Set(
                {"index": "1"},
                sicd.TStart(str(t_start)),
                sicd.TEnd(str(t_end)),
                sicd.IPPStart(str(ipp_start)),
                sicd.IPPEnd(str(ipp_end)),
                sicd.IPPPoly(),
            ),
        ),
    )
    sksicd.PolyType().set_elem(timeline.find("./{*}IPP/{*}Set/{*}IPPPoly"), ipp_poly)

    position = sicd.Position(sicd.ARPPoly())
    sksicd.XyzPolyType().set_elem(position.find("./{*}ARPPoly"), apc_poly)

    radar_collection = sicd.RadarCollection(
        sicd.TxFrequency(sicd.Min(str(tx_freq_min)), sicd.Max(str(tx_freq_max))),
        sicd.Waveform(
            {"size": "1"},
            sicd.WFParameters(
                {"index": "1"},
                sicd.TxPulseLength(str(tx_pulse_length)),
                sicd.TxRFBandwidth(str(tx_rf_bw)),
                sicd.TxFreqStart(str(tx_freq_min)),
                sicd.TxFMRate(str(tx_fm_rate)),
                sicd.RcvDemodType(rcv_demod_type),
                sicd.ADCSampleRate(str(adc_sample_rate)),
                sicd.RcvFMRate(str(0)),
            ),
        ),
        sicd.TxPolarization(tx_polarization),
        sicd.RcvChannels(
            {"size": "1"},
            sicd.ChanParameters(
                {"index": "1"},
                sicd.TxRcvPolarization(tx_rcv_polarization),
            ),
        ),
    )

    image_formation = sicd.ImageFormation(
        sicd.RcvChanProc(sicd.NumChanProc("1"), sicd.ChanIndex("1")),
        sicd.TxRcvPolarizationProc(tx_rcv_polarization_proc),
        sicd.TStartProc(str(t_start_proc)),
        sicd.TEndProc(str(t_end_proc)),
        sicd.TxFrequencyProc(
            sicd.MinProc(str(tx_freq_proc[0])), sicd.MaxProc(str(tx_freq_proc[1]))
        ),
        sicd.ImageFormAlgo(image_form_algo),
        sicd.STBeamComp(st_beam_comp),
        sicd.ImageBeamComp(image_beam_comp),
        sicd.AzAutofocus(az_autofocus),
        sicd.RgAutofocus(rg_autofocus),
    )

    radiometric = sicd.Radiometric(
        sicd.BetaZeroSFPoly(),
    )
    sksicd.Poly2dType().set_elem(
        radiometric.find("./{*}BetaZeroSFPoly"), beta_zero_sf_poly
    )

    rma = sicd.RMA(
        sicd.RMAlgoType(rma_algo_type),
        sicd.ImageType(image_type),
        sicd.INCA(
            sicd.TimeCAPoly(),
            sicd.R_CA_SCP(str(r_ca_scp)),
            sicd.FreqZero(str(freq_zero)),
            sicd.DRateSFPoly(),
            sicd.DopCentroidPoly(),
            sicd.DopCentroidCOA(dop_centroid_coa),
        ),
    )
    sksicd.PolyType().set_elem(rma.find("./{*}INCA/{*}TimeCAPoly"), time_ca_poly)
    sksicd.Poly2dType().set_elem(rma.find("./{*}INCA/{*}DRateSFPoly"), dr_sf_poly)
    sksicd.Poly2dType().set_elem(
        rma.find("./{*}INCA/{*}DopCentroidPoly"), dop_centroid_poly
    )

    sicd_xml_obj = sicd.SICD(
        collection_info,
        image_creation,
        image_data,
        geo_data,
        grid,
        timeline,
        position,
        radar_collection,
        image_formation,
        rma,
    )

    scp_coa = sksicd.compute_scp_coa(sicd_xml_obj.getroottree())
    sicd_xml_obj = sicd.SICD(
        collection_info,
        image_creation,
        image_data,
        geo_data,
        grid,
        timeline,
        position,
        radar_collection,
        image_formation,
        scp_coa,
        radiometric,
        rma,
    )

    new_radiometric = _update_radiometric_node(sicd_xml_obj.getroottree())
    sicd_xml_obj = sicd.SICD(
        collection_info,
        image_creation,
        image_data,
        geo_data,
        grid,
        timeline,
        position,
        radar_collection,
        image_formation,
        scp_coa,
        new_radiometric,
        rma,
    )

    sicd_xmltree = sicd_xml_obj.getroottree()

    # Update ImageCorners
    image_grid_locations = (
        np.array(
            [[0, 0], [0, num_cols - 1], [num_rows - 1, num_cols - 1], [num_rows - 1, 0]]
        )
        - scp_pixel
    ) * [row_ss, col_ss]
    icp_ecef, _, _ = sksicd.image_to_ground_plane(
        sicd_xmltree,
        image_grid_locations,
        scp_ecf,
        sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
    )
    icp_llh = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
    xml_helper = sksicd.XmlHelper(sicd_xmltree)
    xml_helper.set("./{*}GeoData/{*}ImageCorners", icp_llh[:, :2])

    # Check for XML consistency
    sicd_con = SicdConsistency(sicd_xmltree)
    sicd_con.check()
    sicd_con.print_result(fail_detail=True)

    # Grab the data
    real_part = h5_attrs["s_i"]
    imag_part = h5_attrs["s_q"]
    complex_data_arr = np.dstack((real_part, imag_part))
    dtype = complex_data_arr.dtype
    view_dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(dtype.byteorder)
    complex_data_arr = complex_data_arr.view(dtype=view_dtype).reshape(
        complex_data_arr.shape[:2]
    )

    complex_data_arr = np.transpose(complex_data_arr)
    if look > 0:
        complex_data_arr = np.fliplr(complex_data_arr)

    metadata = sksicd.NitfMetadata(
        xmltree=sicd_xmltree,
        file_header_part={
            "ostaid": ostaid,
            "ftitle": core_name,
            "security": {
                "clas": classification[0].upper(),
                "clsy": "US",
            },
        },
        im_subheader_part={
            "iid2": core_name,
            "security": {
                "clas": classification[0].upper(),
                "clsy": "US",
            },
            "isorce": collector_name,
        },
        de_subheader_part={
            "security": {
                "clas": classification[0].upper(),
                "clsy": "US",
            },
        },
    )

    with sicd_filename.open("wb") as f:
        with sksicd.NitfWriter(f, metadata) as writer:
            writer.write_image(complex_data_arr)


def main(args=None):
    """CLI for converting Iceye SLC to SICD"""
    parser = argparse.ArgumentParser(
        description="Converts an Iceye HDF5 file into a SICD.",
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_h5_file",
        type=pathlib.Path,
        help="path of the input HDF5 file",
    )
    parser.add_argument(
        "classification",
        type=str,
        help="content of the /SICD/CollectionInfo/Classification node in the SICD XML",
    )
    parser.add_argument(
        "output_sicd_file",
        type=pathlib.Path,
        help="path of the output SICD file",
    )
    parser.add_argument(
        "--ostaid",
        type=str,
        help="content of the originating station ID (OSTAID) field of the NITF header",
        default="Unknown",
    )
    config = parser.parse_args(args)

    hdf5_to_sicd(
        config.input_h5_file,
        config.output_sicd_file,
        classification=config.classification,
        ostaid=config.ostaid,
    )


if __name__ == "__main__":
    main()
