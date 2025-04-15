"""
===================
CSK Complex to SICD
===================

Convert a complex image from the Cosmo-SkyMed HD5 SLC into SICD 1.4

During development, the following documents were considered:

"COSMO-SkyMed Seconda Generazione: System and Products Description", Revision A
"COSMO-SkyMed Mission and Products Description", Issue 2

In addition, SARPy was consulted on how to use the CSK/CSG to compute SICD
metadata that would predict the complex data characteristics

"""

import argparse
import contextlib
import pathlib

import dateutil.parser
import h5py
import lxml.builder
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import sarkit.verification
import sarkit.wgs84
import scipy.constants
import scipy.optimize

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}

MODE_TYPE_MAP = {
    # CSK
    "HIMAGE": "STRIPMAP",
    "PINGPONG": "STRIPMAP",
    "WIDEREGION": "STRIPMAP",
    "HUGEREGION": "STRIPMAP",
    "ENHANCED SPOTLIGHT": "DYNAMIC STRIPMAP",
    "SMART": "DYNAMIC STRIPMAP",
    # CSG
    "STRIPMAP": "STRIPMAP",
    "QUADPOL": "STRIPMAP",
    # KOMPSAT-5 (rebranded CSK)
    "STANDARD": "STRIPMAP",
    "ENHANCED STANDARD": "STRIPMAP",
    "WIDE SWATH": "STRIPMAP",
    "ENHANCED WIDE SWATH": "STRIPMAP",
    "HIGH RESOLUTION": "DYNAMIC STRIPMAP",
    "ENHANCED HIGH RESOLUTION": "DYNAMIC STRIPMAP",
    "ULTRA HIGH RESOLUTION": "DYNAMIC STRIPMAP",
}

PIXEL_TYPE_MAP = {
    "float32": "RE32F_IM32F",
    "int16": "RE16I_IM16I",
}


def try_decode(value):
    """Try to decode a value.

    `h5py` attributes are arrays.  Strings get returned as `bytes` and need to be decoded.
    """
    with contextlib.suppress(AttributeError):
        return value.decode()
    return value


def extract_attributes(item, add_dataset_info=False):
    """Recursively extracts a nested dictionary of attributes from an item and its descendants"""
    retval = dict()
    if add_dataset_info:
        if hasattr(item, "shape"):
            retval["__shape__"] = item.shape
        if hasattr(item, "dtype"):
            retval["__dtype__"] = str(item.dtype)
    if hasattr(item, "items"):
        for name, contents in item.items():
            retval[name] = extract_attributes(contents, add_dataset_info)
    retval.update(sorted((key, try_decode(value)) for key, value in item.attrs.items()))
    return retval


def fit_state_vectors(
    fit_time_range, times, positions, velocities=None, accelerations=None, order=5
):
    """Fit a polynomial to time-stamped state vectors

    Will only use the minimum number of state vectors to get enough data for a
    polynomial of order ``order`` or to span fit_time_range, whichever is more.

    Args
    ----
    fit_time_range: 2-tuple of float
        start and end time of the fit
    times: array-like
        Time stamp for each state vector
    positions: array-like
        Position in each state vector
    velocities: array-like
        Velocity in each state vector
    accelerations: array-like
        Acceleration in each state vector
    order: int
        Order of the polynomial fit.

    Returns
    -------
    Position vs time Polynomial
    """
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


def compute_apc_poly(h5_attrs, ref_time, start_time, stop_time):
    """Creates an Aperture Phase Center (APC) poly that orbits the Earth above the equator.

    Polynomial generates 3D coords in ECF as a function of time from start of collect.

    Parameters
    ----------
    h5_attrs: dict
        The collection metadata
    ref_time: datetime.datetime
        The time at which the orbit goes through the `apc_pos`.
    start_time: datetime.datetime
        The start time to fit.
    stop_time: datetime.datetime
        The end time to fit.

    Returns
    -------
    `numpy.ndarray`, shape=(6, 3)
        APC poly
    """
    position = h5_attrs["ECEF Satellite Position"]
    velocity = h5_attrs["ECEF Satellite Velocity"]
    acceleration = h5_attrs["ECEF Satellite Acceleration"]
    times = h5_attrs["State Vectors Times"] - (start_time - ref_time).total_seconds()
    apc_poly = fit_state_vectors(
        (0, (stop_time - start_time).total_seconds()),
        times,
        position,
        velocity,
        acceleration,
        order=5,
    )

    return apc_poly


def polyfit2d(x, y, z, deg1, deg2):
    """Fits 2d polynomials to data.

    Example
    -------
    >>> x, y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-2, 2, 53), indexing='ij')
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


def hdf5_to_sicd(h5_filename, sicd_filename, classification, ostaid):
    with h5py.File(h5_filename, "r") as h5file:
        h5_attrs = extract_attributes(h5file)
        mission_id = h5_attrs["Mission ID"]
        if mission_id == "CSG":
            dataset_str = "IMG"
        else:
            dataset_str = "SBI"
        sample_data_h5_path = f"S01/{dataset_str}"
        sample_data_shape = h5file[sample_data_h5_path].shape
        sample_data_dtype = h5file[sample_data_h5_path].dtype

    # Timeline
    ref_time = dateutil.parser.parse(h5_attrs["Reference UTC"])
    collection_start_time = dateutil.parser.parse(h5_attrs["Scene Sensing Start UTC"])
    collection_stop_time = dateutil.parser.parse(h5_attrs["Scene Sensing Stop UTC"])
    collection_duration = (collection_stop_time - collection_start_time).total_seconds()
    prf = h5_attrs["S01"]["PRF"]
    num_pulses = int(np.ceil(collection_duration * prf))
    look = {"LEFT": 1, "RIGHT": -1}[h5_attrs["Look Side"]]

    # Collection Info
    collector_name = h5_attrs["Satellite ID"]
    date_str = collection_start_time.strftime("%d%b%y").upper()
    time_str = collection_start_time.strftime("%H%M%S") + "Z"
    core_name = f"{date_str}_{h5_attrs['Satellite ID']}_{time_str}"
    acquisition_mode = h5_attrs["Acquisition Mode"]
    if "scan" in acquisition_mode.lower():
        raise ValueError("ScanSar modes not supported")
    radar_mode_type = MODE_TYPE_MAP.get(acquisition_mode, None)
    if not radar_mode_type:
        radar_mode_type = "DYNAMIC STRIPMAP"
    radar_mode_id = h5_attrs["Multi-Beam ID"]

    # Creation Info
    creation_time = dateutil.parser.parse(h5_attrs["Product Generation UTC"])
    creation_site = h5_attrs["Processing Centre"]
    l0_ver = h5_attrs.get("L0 Software Version", "NONE")
    l1_ver = h5_attrs.get("L1A Software Version", "NONE")
    creation_application = f"L0: {l0_ver}, L1: {l1_ver}"

    # Image Data
    pixel_type = PIXEL_TYPE_MAP[sample_data_dtype.name]
    num_rows = sample_data_shape[1]
    num_cols = sample_data_shape[0]
    first_row = 0
    first_col = 0
    scp_pixel = np.array([num_rows // 2, num_cols // 2])

    # Position
    apc_poly = compute_apc_poly(
        h5_attrs, ref_time, collection_start_time, collection_stop_time
    )

    # Radar Collection
    center_frequency = h5_attrs["Radar Frequency"]
    tx_pulse_length = h5_attrs["S01"]["Range Chirp Length"]
    tx_fm_rate = h5_attrs["S01"]["Range Chirp Rate"]
    tx_rf_bw = np.abs(tx_fm_rate * tx_pulse_length)
    tx_freq_min = center_frequency - 0.5 * tx_rf_bw
    tx_freq_max = center_frequency + 0.5 * tx_rf_bw
    tx_freq_start = center_frequency - (tx_pulse_length / 2 * tx_fm_rate)
    adc_sample_rate = h5_attrs["S01"]["Sampling Rate"]
    rcv_window_length = h5_attrs["S01"]["Echo Sampling Window Length"] / adc_sample_rate
    if mission_id == "CSG":
        tx_polarization = h5_attrs["Polarization"][0]
        rcv_polarization = h5_attrs["Polarization"][1]
    else:
        tx_polarization = h5_attrs["S01"]["Polarisation"][0]
        rcv_polarization = h5_attrs["S01"]["Polarisation"][1]
    tx_rcv_polarization = f"{tx_polarization}:{rcv_polarization}"

    # Grid
    assert h5_attrs["Lines Order"] == "EARLY-LATE"
    assert h5_attrs["Columns Order"] == "NEAR-FAR"
    spacings = np.array(
        [
            h5_attrs["S01"][dataset_str]["Column Spacing"],
            h5_attrs["S01"][dataset_str]["Line Spacing"],
        ]
    )
    intervals = np.array(
        [
            h5_attrs["S01"][dataset_str]["Column Time Interval"],
            h5_attrs["S01"][dataset_str]["Line Time Interval"],
        ]
    )
    zd_az_0 = h5_attrs["S01"][dataset_str]["Zero Doppler Azimuth First Time"]
    zd_rg_0 = h5_attrs["S01"][dataset_str]["Zero Doppler Range First Time"]
    row_bw = (
        h5_attrs["S01"]["Range Focusing Bandwidth"] * 2 / scipy.constants.speed_of_light
    )
    row_wid = 1 / row_bw
    col_bw = (
        min(h5_attrs["S01"]["Azimuth Focusing Transition Bandwidth"] * intervals[1], 1)
        / spacings[1]
    )
    col_wid = 1 / col_bw

    # CA and COA times
    num_grid_pts = 51
    grid_indices = np.stack(
        np.meshgrid(
            np.linspace(0, num_rows - 1, num_grid_pts),
            np.linspace(0, num_cols - 1, num_grid_pts),
            indexing="ij",
        ),
        axis=-1,
    )
    grid_coords = (grid_indices - scp_pixel) * spacings
    time_coords = grid_indices[:, ::-look] * intervals + np.array([zd_rg_0, zd_az_0])
    start_minus_ref = (collection_start_time - ref_time).total_seconds()

    if mission_id == "CSG":
        range_ref = h5_attrs["S01"]["Range Polynomial Reference Time"]
        azimuth_ref = h5_attrs["S01"]["Azimuth Polynomial Reference Time"]
        azimuth_ref_zd = h5_attrs["S01"]["Azimuth Polynomial Reference Time - ZD"]
        azimuth_first_time = h5_attrs["S01"]["B0001"]["Azimuth First Time"]
        azimuth_last_time = h5_attrs["S01"]["B0001"]["Azimuth Last Time"]
        raw_times = np.linspace(azimuth_first_time, azimuth_last_time, num_grid_pts)

        centroid_range_poly = h5_attrs["S01"][
            "Doppler Centroid vs Range Time Polynomial"
        ]
        centroid_azimuth_poly = h5_attrs["S01"][
            "Doppler Centroid vs Azimuth Time Polynomial - RAW"
        ]
        raw_doppler_centroid = npp.polyval(
            raw_times - azimuth_ref, centroid_azimuth_poly
        )

        rate_range_poly = h5_attrs["S01"]["Doppler Rate vs Range Time Polynomial"]
        rate_azimuth_poly = h5_attrs["S01"]["Doppler Rate vs Azimuth Time Polynomial"]
        raw_doppler_rate = npp.polyval(raw_times - azimuth_ref, rate_azimuth_poly)

        zd_times = raw_times - raw_doppler_centroid / raw_doppler_rate

        zd_to_az_centroid = npp.polyfit(
            zd_times - azimuth_ref_zd, raw_doppler_centroid, 4
        )
        doppler_centroid = (
            npp.polyval(time_coords[..., 0] - range_ref, centroid_range_poly)
            + npp.polyval(time_coords[..., 1] - azimuth_ref_zd, zd_to_az_centroid)
            - (centroid_range_poly[0] + zd_to_az_centroid[0]) / 2
        )

        zd_to_az_rate = npp.polyfit(zd_times - azimuth_ref_zd, raw_doppler_rate, 4)
        doppler_rate = (
            npp.polyval(time_coords[..., 0] - range_ref, rate_range_poly)
            + npp.polyval(time_coords[..., 1] - azimuth_ref_zd, zd_to_az_rate)
            - (rate_range_poly[0] + zd_to_az_rate[0]) / 2
        )
    else:
        range_ref = h5_attrs["Range Polynomial Reference Time"]
        azimuth_ref = h5_attrs["Azimuth Polynomial Reference Time"]

        centroid_range_poly = h5_attrs["Centroid vs Range Time Polynomial"]
        centroid_azimuth_poly = h5_attrs["Centroid vs Azimuth Time Polynomial"]
        doppler_centroid = (
            npp.polyval(time_coords[..., 0] - range_ref, centroid_range_poly)
            + npp.polyval(time_coords[..., 1] - azimuth_ref, centroid_azimuth_poly)
            - (centroid_range_poly[0] + centroid_azimuth_poly[0]) / 2
        )

        rate_range_poly = h5_attrs["Doppler Rate vs Range Time Polynomial"]
        rate_azimuth_poly = h5_attrs["Doppler Rate vs Azimuth Time Polynomial"]
        doppler_rate = (
            npp.polyval(time_coords[..., 0] - range_ref, rate_range_poly)
            + npp.polyval(time_coords[..., 1] - azimuth_ref, rate_azimuth_poly)
            - (rate_range_poly[0] + rate_azimuth_poly[0]) / 2
        )

    range_rate_per_hz = -scipy.constants.speed_of_light / (2 * center_frequency)
    range_rate = doppler_centroid * range_rate_per_hz
    range_rate_rate = doppler_rate * range_rate_per_hz
    doppler_centroid_poly = polyfit2d(
        grid_coords[..., 0].flatten(),
        grid_coords[..., 1].flatten(),
        doppler_centroid.flatten(),
        4,
        4,
    )
    doppler_rate_poly = polyfit2d(
        grid_coords[..., 0].flatten(),
        grid_coords[..., 1].flatten(),
        doppler_rate.flatten(),
        4,
        4,
    )
    time_ca_samps = time_coords[..., 1] - start_minus_ref
    time_ca_poly = npp.polyfit(
        grid_coords[..., 1].flatten(), time_ca_samps.flatten(), 1
    )
    time_coa_samps = time_ca_samps + range_rate / range_rate_rate
    time_coa_poly = polyfit2d(
        grid_coords[..., 0].flatten(),
        grid_coords[..., 1].flatten(),
        time_coa_samps.flatten(),
        4,
        4,
    )

    range_ca = time_coords[..., 0] * scipy.constants.speed_of_light / 2
    speed_ca = npl.norm(
        npp.polyval(time_coords[..., 1] - start_minus_ref, npp.polyder(apc_poly)),
        axis=0,
    )
    drsf = range_rate_rate * range_ca / speed_ca**2
    drsf_poly = polyfit2d(
        grid_coords[..., 0].flatten(),
        grid_coords[..., 1].flatten(),
        drsf.flatten(),
        4,
        4,
    )

    llh_ddm = h5_attrs["Scene Centre Geodetic Coordinates"]
    scp_drsf = drsf_poly[0, 0]
    scp_tca = time_ca_poly[0]
    scp_rca = (
        (zd_rg_0 + scp_pixel[0] * intervals[0]) * scipy.constants.speed_of_light / 2
    )
    scp_tcoa = time_coa_poly[0, 0]
    scp_delta_t_coa = scp_tcoa - scp_tca
    scp_varp_ca_mag = npl.norm(npp.polyval(scp_tca, npp.polyder(apc_poly)))
    scp_rcoa = np.sqrt(scp_rca**2 + scp_drsf * scp_varp_ca_mag**2 * scp_delta_t_coa**2)
    scp_rratecoa = scp_drsf / scp_rcoa * scp_varp_ca_mag**2 * scp_delta_t_coa

    def obj(hae):
        scene_pos = sarkit.wgs84.geodetic_to_cartesian([llh_ddm[0], llh_ddm[1], hae])
        delta_t = np.linspace(-0.01, 0.01)
        arp_pos = npp.polyval(scp_tca + delta_t, apc_poly).T
        arp_speed = npl.norm(npp.polyval(scp_tca, npp.polyder(apc_poly)), axis=0)
        range_ = npl.norm(arp_pos - scene_pos, axis=1)
        range_poly = npp.polyfit(delta_t, range_, len(apc_poly))
        test_drsf = 2 * range_poly[2] * range_poly[0] / arp_speed**2
        return scp_drsf - test_drsf

    scp_hae = scipy.optimize.brentq(obj, -30e3, 30e3)
    sc_ecf = sarkit.wgs84.geodetic_to_cartesian(llh_ddm)
    scp_set = sksicd.projection.ProjectionSets(
        t_COA=np.array([scp_tcoa]),
        ARP_COA=np.array([npp.polyval(scp_tcoa, apc_poly)]),
        VARP_COA=np.array([npp.polyval(scp_tcoa, npp.polyder(apc_poly))]),
        R_COA=np.array([scp_rcoa]),
        Rdot_COA=np.array([scp_rratecoa]),
    )
    scp_ecf, _, _ = sksicd.projection.r_rdot_to_constant_hae_surface(
        look, sc_ecf, "MONOSTATIC", scp_set, scp_hae
    )
    scp_ecf = scp_ecf[0]
    scp_llh = sarkit.wgs84.cartesian_to_geodetic(scp_ecf)
    scp_ca_pos = npp.polyval(scp_tca, apc_poly)
    scp_ca_vel = npp.polyval(scp_tcoa, npp.polyder(apc_poly))
    los = scp_ecf - scp_ca_pos
    u_row = los / npl.norm(los)
    left = np.cross(scp_ca_pos, scp_ca_vel)
    look = np.sign(np.dot(left, u_row))
    spz = -look * np.cross(u_row, scp_ca_vel)
    uspz = spz / npl.norm(spz)
    u_col = np.cross(uspz, u_row)

    # Build XML
    sicd = lxml.builder.ElementMaker(
        namespace=NSMAP["sicd"], nsmap={None: NSMAP["sicd"]}
    )
    collection_info = sicd.CollectionInfo(
        sicd.CollectorName(collector_name),
        sicd.CoreName(core_name),
        sicd.CollectType("MONOSTATIC"),
        sicd.RadarMode(sicd.ModeType(radar_mode_type), sicd.ModeID(radar_mode_id)),
        sicd.Classification(classification),
    )
    image_creation = sicd.ImageCreation(
        sicd.Application(creation_application),
        sicd.DateTime(creation_time.isoformat() + "Z"),
        sicd.Site(creation_site),
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

    # Placeholder locations
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

    dc_sgn = np.sign(-doppler_rate_poly[0, 0])
    col_deltakcoa_poly = (
        -look * dc_sgn * doppler_centroid_poly * intervals[1] / spacings[1]
    )
    vertices = [
        (0, 0),
        (0, num_cols - 1),
        (num_rows - 1, num_cols - 1),
        (num_rows - 1, 0),
    ]
    coords = (vertices - scp_pixel) * spacings
    deltaks = npp.polyval2d(coords[:, 0], coords[:, 1], col_deltakcoa_poly)
    dk1 = deltaks.min() - col_bw / 2
    dk2 = deltaks.max() + col_bw / 2
    if dk1 < -0.5 / spacings[1] or dk2 > 0.5 / spacings[1]:
        dk1 = -0.5 / spacings[1]
        dk2 = -dk1

    row_window_name = h5_attrs["Range Focusing Weighting Function"]
    row_window_coeff = h5_attrs["Range Focusing Weighting Coefficient"]
    col_window_name = h5_attrs["Azimuth Focusing Weighting Function"]
    col_window_coeff = h5_attrs["Azimuth Focusing Weighting Coefficient"]

    grid = sicd.Grid(
        sicd.ImagePlane("SLANT"),
        sicd.Type("RGZERO"),
        sicd.TimeCOAPoly(),
        sicd.Row(
            sicd.UVectECF(*make_xyz(u_row)),
            sicd.SS(str(spacings[0])),
            sicd.ImpRespWid(str(row_wid)),
            sicd.Sgn("-1"),
            sicd.ImpRespBW(str(row_bw)),
            sicd.KCtr(str(center_frequency / (scipy.constants.speed_of_light / 2))),
            sicd.DeltaK1(str(-row_bw / 2)),
            sicd.DeltaK2(str(row_bw / 2)),
            sicd.DeltaKCOAPoly(),
            sicd.WgtType(
                sicd.WindowName(row_window_name),
                sicd.Parameter({"name": "COEFFICIENT"}, str(row_window_coeff)),
            ),
        ),
        sicd.Col(
            sicd.UVectECF(*make_xyz(u_col)),
            sicd.SS(str(spacings[1])),
            sicd.ImpRespWid(str(col_wid)),
            sicd.Sgn("-1"),
            sicd.ImpRespBW(str(col_bw)),
            sicd.KCtr("0"),
            sicd.DeltaK1(str(dk1)),
            sicd.DeltaK2(str(dk2)),
            sicd.DeltaKCOAPoly(),
            sicd.WgtType(
                sicd.WindowName(col_window_name),
                sicd.Parameter({"name": "COEFFICIENT"}, str(col_window_coeff)),
            ),
        ),
    )
    sksicd.Poly2dType().set_elem(grid.find("./{*}TimeCOAPoly"), time_coa_poly)
    sksicd.Poly2dType().set_elem(grid.find("./{*}Row/{*}DeltaKCOAPoly"), [[0]])
    sksicd.Poly2dType().set_elem(
        grid.find("./{*}Col/{*}DeltaKCOAPoly"), col_deltakcoa_poly
    )
    if row_window_name == "HAMMING":
        wgts = scipy.signal.windows.general_hamming(512, row_window_coeff, sym=True)
        wgtfunc = sicd.WgtFunct()
        sksicd.TRANSCODERS["Grid/Row/WgtFunct"].set_elem(wgtfunc, wgts)
        grid.find("./{*}Row").append(wgtfunc)
        row_broadening_factor = broadening_from_amp(wgts)
        row_wid = row_broadening_factor / row_bw
        sksicd.DblType().set_elem(grid.find("./{*}Row/{*}ImpRespWid"), row_wid)
    if col_window_name == "HAMMING":
        wgts = scipy.signal.windows.general_hamming(512, col_window_coeff, sym=True)
        wgtfunc = sicd.WgtFunct()
        sksicd.TRANSCODERS["Grid/Col/WgtFunct"].set_elem(wgtfunc, wgts)
        grid.find("./{*}Col").append(wgtfunc)
        col_broadening_factor = broadening_from_amp(wgts)
        col_wid = col_broadening_factor / col_bw
        sksicd.DblType().set_elem(grid.find("./{*}Col/{*}ImpRespWid"), col_wid)

    timeline = sicd.Timeline(
        sicd.CollectStart(collection_start_time.isoformat() + "Z"),
        sicd.CollectDuration(str(collection_duration)),
        sicd.IPP(
            {"size": "1"},
            sicd.Set(
                {"index": "1"},
                sicd.TStart(str(0)),
                sicd.TEnd(str(num_pulses / prf)),
                sicd.IPPStart(str(0)),
                sicd.IPPEnd(str(num_pulses - 1)),
                sicd.IPPPoly(),
            ),
        ),
    )
    sksicd.PolyType().set_elem(timeline.find("./{*}IPP/{*}Set/{*}IPPPoly"), [0, prf])

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
                sicd.TxFreqStart(str(tx_freq_start)),
                sicd.TxFMRate(str(tx_fm_rate)),
                sicd.RcvWindowLength(str(rcv_window_length)),
                sicd.ADCSampleRate(str(adc_sample_rate)),
            ),
        ),
        sicd.TxPolarization(tx_polarization),
        sicd.RcvChannels(
            {"size": "1"},
            sicd.ChanParameters(
                {"index": "1"}, sicd.TxRcvPolarization(tx_rcv_polarization)
            ),
        ),
    )

    image_formation = sicd.ImageFormation(
        sicd.RcvChanProc(sicd.NumChanProc("1"), sicd.ChanIndex("1")),
        sicd.TxRcvPolarizationProc(tx_rcv_polarization),
        sicd.TStartProc(str(0)),
        sicd.TEndProc(str(collection_duration)),
        sicd.TxFrequencyProc(
            sicd.MinProc(str(tx_freq_min)), sicd.MaxProc(str(tx_freq_max))
        ),
        sicd.ImageFormAlgo("RMA"),
        sicd.STBeamComp("NO"),
        sicd.ImageBeamComp("SV"),
        sicd.AzAutofocus("NO"),
        sicd.RgAutofocus("NO"),
    )

    rma = sicd.RMA(
        sicd.RMAlgoType("OMEGA_K"),
        sicd.ImageType("INCA"),
        sicd.INCA(
            sicd.TimeCAPoly(),
            sicd.R_CA_SCP(str(scp_rca)),
            sicd.FreqZero(str(center_frequency)),
            sicd.DRateSFPoly(),
            sicd.DopCentroidPoly(),
        ),
    )
    sksicd.PolyType().set_elem(rma.find("./{*}INCA/{*}TimeCAPoly"), time_ca_poly)
    sksicd.Poly2dType().set_elem(rma.find("./{*}INCA/{*}DRateSFPoly"), drsf_poly)
    sksicd.Poly2dType().set_elem(
        rma.find("./{*}INCA/{*}DopCentroidPoly"), doppler_centroid_poly
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

    image_formation.addnext(sksicd.compute_scp_coa(sicd_xml_obj.getroottree()))

    # Add Radiometric
    if mission_id == "CSK":
        if h5_attrs["Range Spreading Loss Compensation Geometry"] != "NONE":
            slant_range = h5_attrs["Reference Slant Range"]
            exp = h5_attrs["Reference Slant Range Exponent"]
            scale_factor = slant_range ** (2 * exp)
            rescale_factor = h5_attrs["Rescaling Factor"]
            scale_factor /= rescale_factor * rescale_factor
            if h5_attrs.get("Calibration Constant Compensation Flag", None) == 0:
                cal = h5_attrs["S01"]["Calibration Constant"]
                scale_factor /= cal
            betazero_poly = np.array([[scale_factor]])
            graze = np.deg2rad(float(sicd_xml_obj.findtext("./{*}SCPCOA/{*}GrazeAng")))
            twist = np.deg2rad(float(sicd_xml_obj.findtext("./{*}SCPCOA/{*}TwistAng")))
            sigmazero_poly = betazero_poly * np.cos(graze) * np.cos(twist)
            gammazero_poly = betazero_poly / np.tan(graze) * np.cos(twist)
            radiometric = sicd.Radiometric(
                sicd.SigmaZeroSFPoly(), sicd.BetaZeroSFPoly(), sicd.GammaZeroSFPoly()
            )
            sksicd.Poly2dType().set_elem(
                radiometric.find("./{*}SigmaZeroSFPoly"), sigmazero_poly
            )
            sksicd.Poly2dType().set_elem(
                radiometric.find("./{*}BetaZeroSFPoly"), betazero_poly
            )
            sksicd.Poly2dType().set_elem(
                radiometric.find("./{*}GammaZeroSFPoly"), gammazero_poly
            )
            sicd_xml_obj.find("./{*}RMA").addprevious(radiometric)

    # Add Geodata Corners
    sicd_xmltree = sicd_xml_obj.getroottree()
    image_grid_locations = (
        np.array(
            [[0, 0], [0, num_cols - 1], [num_rows - 1, num_cols - 1], [num_rows - 1, 0]]
        )
        - scp_pixel
    ) * spacings
    icp_ecef, _, _ = sksicd.image_to_ground_plane(
        sicd_xmltree,
        image_grid_locations,
        scp_ecf,
        sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
    )
    icp_llh = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
    xml_helper = sksicd.XmlHelper(sicd_xmltree)
    xml_helper.set("./{*}GeoData/{*}ImageCorners", icp_llh[:, :2])

    # Validate XML
    sicd_con = sarkit.verification.SicdConsistency(sicd_xmltree)
    sicd_con.check()
    sicd_con.print_result(fail_detail=True)

    # Grab the data
    with h5py.File(h5_filename, "r") as h5file:
        data_arr = np.asarray(h5file[sample_data_h5_path])
        dtype = data_arr.dtype
        view_dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(
            dtype.byteorder
        )
        complex_data_arr = np.squeeze(data_arr.view(view_dtype))
    complex_data_arr = np.transpose(complex_data_arr)
    if look > 0:
        complex_data_arr = complex_data_arr[:, ::-1]

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
            "tgtid": "",
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
    parser = argparse.ArgumentParser(
        description="Converts a CSK SCS HDF5 file into a SICD.",
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_h5_file", type=pathlib.Path, help="path of the input HDF5 file"
    )
    parser.add_argument(
        "classification",
        help="content of the /SICD/CollectionInfo/Classification node in the SICD XML",
    )
    parser.add_argument(
        "output_sicd_file", type=pathlib.Path, help="path of the output SICD file"
    )
    parser.add_argument(
        "--ostaid",
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
