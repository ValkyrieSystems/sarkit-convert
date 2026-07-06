"""
=====================
NISAR Complex to SICD
=====================

Convert a complex image from the NISAR HD5 RSLC into SICD.

Note: In the development of this converter "NISAR_D-102268_RevE_NASA_SDS_Product_Specification_L1_RSLC" was used.
"""

import argparse
import datetime
import pathlib

import dateutil.parser
import h5py
import lxml.etree
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import sarkit.verification
import sarkit.wgs84
import shapely
from sarkit import _constants

from sarkit_convert import __version__
from sarkit_convert import _utils as utils

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}


def _decode_hdf_type(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")  # Decode byte string
    elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
        value = value.astype(str).tolist()  # Handle ndarrays with type string
    elif isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()  # Handle single value arrays
    return value


def _dictify_hdf(h5_obj, max_size=2**20):
    """Recursively convert hdf5 object into dictionary, decoding types where possible."""
    result = {}

    if isinstance(h5_obj, h5py.Dataset):
        if hasattr(h5_obj, "shape"):
            result["__shape__"] = h5_obj.shape
        if hasattr(h5_obj, "dtype"):
            result["__dtype__"] = str(h5_obj.dtype)
        if h5_obj.nbytes < max_size:
            result["__value__"] = _decode_hdf_type(h5_obj[...])
        for attr_key, attr_value in h5_obj.attrs.items():
            assert attr_key != "__value__"
            result[attr_key] = _decode_hdf_type(attr_value)
    elif isinstance(h5_obj, h5py.Group):
        for key, value in h5_obj.items():  # Iterate over keys
            result[key] = _dictify_hdf(value)
    else:  # Attribute
        for key, value in h5_obj.items():  # Iterate over keys
            result[key] = _decode_hdf_type(value)

    return result


def _get_ref_time(units):
    ref_time_lead = "seconds since "
    if not units.startswith(ref_time_lead):
        raise ValueError(f"Units string {units} does not contain a reference time")
    ref_time = dateutil.parser.parse(units.removeprefix(ref_time_lead))
    return ref_time


def compute_apc_poly(h5dict, start_time, stop_time, pad_time=2):
    """Creates an Aperture Phase Center (APC) poly that fits the provided state vectors

    Polynomial generates 3D coords in ECF as a function of time from start of collect.

    Parameters
    ----------
    h5dict : dict
        The collection metadata
    start_time : datetime.datetime
        The start time to fit.
    stop_time : datetime.datetime
        The end time to fit.
    pad_time : float
        Extra time to fit before start_time and after stop_time

    Returns
    -------
    `numpy.ndarray`, shape=(6, 3)
        APC poly
    """
    orbit_dict = h5dict["science"]["LSAR"]["RSLC"]["metadata"]["orbit"]
    times = orbit_dict["time"]["__value__"]
    positions = orbit_dict["position"]["__value__"]
    velocities = orbit_dict["velocity"]["__value__"]
    ref_time = _get_ref_time(orbit_dict["time"]["units"])
    rel_times = orbit_dict["time"]["__value__"]
    state_times = [
        ref_time + datetime.timedelta(seconds=rel_time) for rel_time in rel_times
    ]
    times = [(state_time - start_time).total_seconds() for state_time in state_times]

    apc_poly = utils.fit_state_vectors(
        (-pad_time, (stop_time - start_time).total_seconds() + pad_time),
        times,
        positions,
        velocities,
        order=5,
    )

    return apc_poly


def hdf5_to_sicd(h5_filename, sicd_filename, frequency, polarization, classification):
    with h5py.File(h5_filename, "r") as h5file:
        h5dict = _dictify_hdf(h5file)

    freq_str = "frequency" + frequency
    # Timeline
    first_zero_doppler_time = dateutil.parser.parse(
        h5dict["science"]["LSAR"]["identification"]["zeroDopplerStartTime"]["__value__"]
    )
    last_zero_doppler_time = dateutil.parser.parse(
        h5dict["science"]["LSAR"]["identification"]["zeroDopplerEndTime"]["__value__"]
    )
    look = {"left": 1, "right": -1}[
        h5dict["science"]["LSAR"]["identification"]["lookDirection"][
            "__value__"
        ].lower()
    ]

    # Maximum integration time derived from finest resolution,
    # longest imaging range and lowest center frequency
    # max_bw = 0.19
    # f_c = 1.25e9
    # k_ctr = f_c * 2 / speed_of_light
    # max_slant_range = 1e6
    # orbital_speed = 7600
    # integration_time = max_slant_range / orbital_speed * max_bw / k_ctr
    # This is just barely over 3, so rounding up to 4 for protection against parameters that exceed these assumptions
    max_integration_time = 4
    half_integration_adjustment = datetime.timedelta(seconds=max_integration_time / 2)
    collect_start = first_zero_doppler_time - half_integration_adjustment
    collect_stop = last_zero_doppler_time + half_integration_adjustment
    collect_duration = (collect_stop - collect_start).total_seconds()

    # Collection Info
    collector_name = h5dict["science"]["LSAR"]["identification"]["instrumentName"][
        "__value__"
    ]
    core_name = h5dict["science"]["LSAR"]["identification"]["granuleId"]["__value__"]
    radar_mode_id = core_name[31:35]
    radar_mode_type = "STRIPMAP"

    # Creation Info
    creation_application = h5dict["science"]["LSAR"]["RSLC"]["metadata"][
        "processingInformation"
    ]["algorithms"]["softwareVersion"]["__value__"]
    creation_date_time = dateutil.parser.parse(
        h5dict["science"]["LSAR"]["identification"]["processingDateTime"]["__value__"]
    )
    creation_site = h5dict["science"]["LSAR"]["identification"]["processingCenter"][
        "__value__"
    ]

    # Position
    apc_poly = compute_apc_poly(h5dict, collect_start, collect_stop)

    # Radar Collection
    acq_center_frequency = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "acquiredCenterFrequency"
    ]["__value__"]
    acq_rf_bw = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "acquiredRangeBandwidth"
    ]["__value__"]
    acq_tx_freq_min = acq_center_frequency - 0.5 * acq_rf_bw
    acq_tx_freq_max = acq_center_frequency + 0.5 * acq_rf_bw

    tx_polarization = polarization[0]
    rcv_polarization = polarization[1]
    tx_rcv_polarization = f"{tx_polarization}:{rcv_polarization}"

    # Image Data
    complex_data = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][polarization]
    assert complex_data["__dtype__"] == "complex64"
    pixel_type = "RE32F_IM32F"
    num_cols, num_rows = complex_data["__shape__"]
    first_row = 0
    first_col = 0
    scp_pixel = np.array((num_rows // 2, num_cols // 2))

    # Image Formation
    tx_rcv_polarization_proc = tx_rcv_polarization
    t_start_proc = 0
    proc_center_frequency = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "processedCenterFrequency"
    ]["__value__"]
    proc_rg_bw = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "processedRangeBandwidth"
    ]["__value__"]
    proc_az_bw = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "processedAzimuthBandwidth"
    ]["__value__"]
    proc_freq_min = proc_center_frequency - 0.5 * proc_rg_bw
    proc_freq_max = proc_center_frequency + 0.5 * proc_rg_bw

    # Some Grid
    img_zd_dict = h5dict["science"]["LSAR"]["RSLC"]["swaths"]["zeroDopplerTime"]
    img_zd_values = img_zd_dict["__value__"]
    img_zd_epoch = _get_ref_time(img_zd_dict["units"])
    img_zd = img_zd_values + (img_zd_epoch - collect_start).total_seconds()
    img_zd_interval = h5dict["science"]["LSAR"]["RSLC"]["swaths"][
        "zeroDopplerTimeSpacing"
    ]["__value__"]
    img_rg = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str]["slantRange"][
        "__value__"
    ]
    row_ss = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str]["slantRangeSpacing"][
        "__value__"
    ]
    scp_rg = img_rg[scp_pixel[0]]
    scp_zd = img_zd[scp_pixel[1]]

    # Compute scene points
    bounding_polygon = shapely.from_wkt(
        h5dict["science"]["LSAR"]["identification"]["boundingPolygon"]["__value__"]
    )
    bp_ecef = sarkit.wgs84.geodetic_to_cartesian(
        np.array(bounding_polygon.exterior.coords)[:, [1, 0, 2]]
    )
    scene_center_ecef = np.mean(bp_ecef, axis=0)
    scene_center_latlon = sarkit.wgs84.cartesian_to_geodetic(scene_center_ecef)[:2]
    scene_height = np.mean(np.asarray(bounding_polygon.exterior.coords)[:, 2])
    scene_ecf = sarkit.wgs84.geodetic_to_cartesian((*scene_center_latlon, scene_height))

    # TODO Determine if reference terrain heights can be sensibly used here
    # In the datasets provided thus far they've been all zero and am unsure
    # what the following calculations would do with varying heights
    #
    # reference_terrain_height = h5dict["science"]["LSAR"]["RSLC"]["metadata"][
    #    "processingInformation"
    # ]["parameters"]["referenceTerrainHeight"]["__value__"]
    # heights = reference_terrain_height[np.newaxis, :]

    grid_size = 11
    ranges_indices = np.linspace(0, len(img_rg) - 1, num=grid_size, dtype=int)
    times_indices = np.linspace(0, len(img_zd) - 1, num=grid_size, dtype=int)
    ranges = img_rg[ranges_indices][:, np.newaxis]
    times = img_zd[times_indices]
    arps = npp.polyval(times, apc_poly).T[np.newaxis, :, :]
    varps = npp.polyval(times, npp.polyder(apc_poly)).T[np.newaxis, :, :]
    times = times[np.newaxis, :]
    rdots = np.zeros_like(ranges)
    heights = np.full_like(ranges, scene_height)
    scene_sets = sksicd.projection.ProjectionSetsMono(
        t_COA=times, ARP_COA=arps, VARP_COA=varps, R_COA=ranges, Rdot_COA=rdots
    )

    scene_pts_ecef, _, _ = sksicd.projection.r_rdot_to_constant_hae_surface(
        look, scene_ecf, scene_sets, heights
    )

    # CA and COA times
    los = arps - scene_pts_ecef
    r_ca = np.linalg.norm(los, axis=-1)
    ulos = los / r_ca[..., np.newaxis]
    r_dot = np.sum(varps * ulos, axis=-1)
    accel = npp.polyval(times, npp.polyder(apc_poly, m=2)).T[np.newaxis, ...]
    vmag = np.linalg.norm(varps, axis=-1)
    rrdot = np.sum(accel * ulos, axis=-1) + (vmag**2 - r_dot**2) / r_ca
    range_rate_per_hz = -_constants.speed_of_light / (2 * acq_center_frequency)
    doppler_rate = rrdot / range_rate_per_hz
    drsf = rrdot * r_ca / vmag**2
    col_ss = np.mean(vmag) * np.abs(img_zd_interval) * np.mean(drsf)

    doppler_centroid = h5dict["science"]["LSAR"]["RSLC"]["metadata"][
        "processingInformation"
    ]["parameters"][freq_str]["dopplerCentroid"]["__value__"].T
    dc_range = h5dict["science"]["LSAR"]["RSLC"]["metadata"]["processingInformation"][
        "parameters"
    ][freq_str]["slantRange"]["__value__"]
    zdt_dict = h5dict["science"]["LSAR"]["RSLC"]["metadata"]["processingInformation"][
        "parameters"
    ][freq_str]["zeroDopplerTime"]
    zdt_values = zdt_dict["__value__"]
    zdt_epoch = _get_ref_time(zdt_dict["units"])
    dc_zd_times = zdt_values + (zdt_epoch - collect_start).total_seconds()

    start_rg_index = max(np.sum(dc_range < img_rg[0]) - 1, 0)
    end_rg_index = max(np.sum(dc_range < img_rg[-1]) + 1, dc_zd_times.size)
    start_zd_index = max(np.sum(dc_zd_times < 0) - 1, 0)
    end_zd_index = max(np.sum(dc_zd_times < collect_duration) + 1, dc_zd_times.size)

    dc_rg = dc_range[start_rg_index:end_rg_index]
    dc_zd = dc_zd_times[start_zd_index:end_zd_index]
    dc_vals = doppler_centroid[start_rg_index:end_rg_index, start_zd_index:end_zd_index]

    dc_coord_rg = dc_rg - scp_rg
    dc_coord_az = (dc_zd - scp_zd) / img_zd_interval * col_ss
    dc_coord_az = dc_coord_az[::-look]
    dc_grid_coords = np.stack(
        np.meshgrid(
            dc_coord_rg.flatten(),
            dc_coord_az.flatten(),
            indexing="ij",
        ),
        axis=-1,
    )

    doppler_centroid_poly = utils.polyfit2d_tol(
        dc_grid_coords[..., 0].flatten(),
        dc_grid_coords[..., 1].flatten(),
        dc_vals.flatten(),
        4,
        4,
        1e-2,
    )

    img_coord_rg = ranges - scp_rg
    img_coord_az = (times - scp_zd) / img_zd_interval * col_ss
    img_coord_az = img_coord_az[:, ::-look]
    img_grid_coords = np.stack(
        np.meshgrid(
            img_coord_rg.flatten(),
            img_coord_az.flatten(),
            indexing="ij",
        ),
        axis=-1,
    )

    doppler_rate_poly = utils.polyfit2d_tol(
        img_grid_coords[..., 0].flatten(),
        img_grid_coords[..., 1].flatten(),
        doppler_rate.flatten(),
        4,
        4,
        1e-3,
    )

    drsf_poly = utils.polyfit2d_tol(
        img_grid_coords[..., 0].flatten(),
        img_grid_coords[..., 1].flatten(),
        drsf.flatten(),
        4,
        4,
        1e-6,
    )

    time_ca_poly = npp.polyfit(img_grid_coords[0, ..., 1].flatten(), times.flatten(), 1)

    dc_img_grid = npp.polyval2d(
        img_grid_coords[..., 0], img_grid_coords[..., 1], doppler_centroid_poly
    )
    time_deltas = dc_img_grid / doppler_rate
    time_coa = times + time_deltas
    time_coa_poly = utils.polyfit2d_tol(
        img_grid_coords[..., 0].flatten(),
        img_grid_coords[..., 1].flatten(),
        time_coa.flatten(),
        4,
        4,
        1e-3,
    )
    min_tcoa = np.min(time_coa)
    max_tcoa = np.max(time_coa)

    # Some Grid stuff to support finishing timeline
    row_kctr = proc_center_frequency / (_constants.speed_of_light / 2)
    row_imp_res_bw = proc_rg_bw / (_constants.speed_of_light / 2)
    col_imp_res_bw = min(proc_az_bw * img_zd_interval, 1) / col_ss

    # Update and finish Timeline
    integration_time = (
        np.max(img_rg)
        / np.mean(vmag)
        * col_imp_res_bw
        / (row_kctr + row_imp_res_bw / 2)
    )
    new_start_adjust = min_tcoa - integration_time / 2
    collect_start = collect_start + datetime.timedelta(seconds=new_start_adjust)
    collect_duration = max_tcoa - min_tcoa + integration_time
    time_ca_poly[0] -= new_start_adjust
    time_coa_poly[0, 0] -= new_start_adjust
    apc_poly = np.asarray(
        [utils.polyshift(apc_poly[:, ndx], new_start_adjust) for ndx in range(3)]
    ).T

    acq_prf = h5dict["science"]["LSAR"]["RSLC"]["swaths"][freq_str][
        "nominalAcquisitionPRF"
    ]["__value__"]
    num_pulses = int(np.round(collect_duration * acq_prf))
    t_start = 0
    t_end = collect_duration
    t_end_proc = collect_duration
    ipp_start = 0
    ipp_end = int(num_pulses - 1)
    ipp_poly = [0, acq_prf]

    # Geo Data
    scp_tca = time_ca_poly[0]
    scp_tcoa = time_coa_poly[0, 0]
    scp_drsf = drsf_poly[0, 0]
    scp_delta_t_coa = scp_tcoa - scp_tca
    scp_varp_ca_mag = npl.norm(npp.polyval(scp_tca, npp.polyder(apc_poly)))
    scp_rcoa = np.sqrt(scp_rg**2 + scp_drsf * scp_varp_ca_mag**2 * scp_delta_t_coa**2)
    scp_rratecoa = scp_drsf / scp_rcoa * scp_varp_ca_mag**2 * scp_delta_t_coa

    scp_set = sksicd.projection.ProjectionSetsMono(
        t_COA=np.array([scp_tcoa]),
        ARP_COA=np.array([npp.polyval(scp_tcoa, apc_poly)]),
        VARP_COA=np.array([npp.polyval(scp_tcoa, npp.polyder(apc_poly))]),
        R_COA=np.array([scp_rcoa]),
        Rdot_COA=np.array([scp_rratecoa]),
    )
    scp_ecf, _, _ = sksicd.projection.r_rdot_to_constant_hae_surface(
        look, scene_ecf, scp_set, scene_height
    )
    scp_ecf = scp_ecf[0]
    scp_llh = sarkit.wgs84.cartesian_to_geodetic(scp_ecf)
    scp_ca_pos = npp.polyval(scp_tca, apc_poly)
    scp_ca_vel = npp.polyval(scp_tca, npp.polyder(apc_poly))
    los = scp_ecf - scp_ca_pos
    u_row = los / npl.norm(los)
    left = np.cross(scp_ca_pos, scp_ca_vel)
    look = np.sign(np.dot(left, u_row))
    spz = -look * np.cross(u_row, scp_ca_vel)
    uspz = spz / npl.norm(spz)
    u_col = np.cross(uspz, u_row)

    # Finish Grid
    row_sgn = -1
    row_deltak_coa_poly = np.array([[0]])
    col_sgn = -1
    col_kctr = 0
    dc_sgn = np.sign(-doppler_rate_poly[0, 0])
    col_deltak_coa_poly = (
        -look * dc_sgn * doppler_centroid_poly * img_zd_interval / col_ss
    )

    vertices = [
        (0, 0),
        (0, num_cols - 1),
        (num_rows - 1, num_cols - 1),
        (num_rows - 1, 0),
    ]
    coords = (vertices - scp_pixel) * np.array([row_ss, col_ss])
    deltaks = npp.polyval2d(coords[:, 0], coords[:, 1], col_deltak_coa_poly)
    dk1 = deltaks.min() - col_imp_res_bw / 2
    dk2 = deltaks.max() + col_imp_res_bw / 2
    if dk1 < -0.5 / col_ss or dk2 > 0.5 / col_ss:
        dk1 = -0.5 / col_ss
        dk2 = -dk1

    row_weighting = h5dict["science"]["LSAR"]["RSLC"]["metadata"][
        "processingInformation"
    ]["parameters"]["rangeChirpWeighting"]
    col_weighting = h5dict["science"]["LSAR"]["RSLC"]["metadata"][
        "processingInformation"
    ]["parameters"]["azimuthChirpWeighting"]
    row_wgts = row_weighting["__value__"]
    row_wgts = row_wgts[row_wgts > 0]
    row_win_name = row_weighting["window_name"]
    row_win_shape = row_weighting["window_shape"]
    col_wgts = col_weighting["__value__"]
    col_wgts = col_wgts[col_wgts > 0]
    row_broadening_factor = utils.broadening_from_amp(row_wgts)
    col_broadening_factor = utils.broadening_from_amp(col_wgts)
    row_imp_res_wid = row_broadening_factor / row_imp_res_bw
    col_imp_res_wid = col_broadening_factor / col_imp_res_bw

    # Build XML
    sicd_xml_obj = lxml.etree.Element(
        f"{{{NSMAP['sicd']}}}SICD", nsmap={None: NSMAP["sicd"]}
    )
    sicd_ew = sksicd.ElementWrapper(sicd_xml_obj)
    sicd_ew["CollectionInfo"] = {
        "CollectorName": collector_name,
        "CoreName": core_name,
        "CollectType": "MONOSTATIC",
        "RadarMode": {
            "ModeType": radar_mode_type,
            "ModeID": radar_mode_id,
        },
        "Classification": classification,
    }
    sicd_ew["ImageCreation"] = {
        "Application": creation_application,
        "DateTime": creation_date_time,
    }
    sicd_ew["ImageData"] = {
        "PixelType": pixel_type,
        "NumRows": num_rows,
        "NumCols": num_cols,
        "FirstRow": first_row,
        "FirstCol": first_col,
        "FullImage": {
            "NumRows": num_rows,
            "NumCols": num_cols,
        },
        "SCPPixel": scp_pixel,
    }

    sicd_ew["GeoData"] = {
        "EarthModel": "WGS_84",
        "SCP": {
            "ECF": scp_ecf,
            "LLH": scp_llh,
        },
    }

    sicd_ew["Grid"] = {
        "ImagePlane": "SLANT",
        "Type": "RGZERO",
        "TimeCOAPoly": time_coa_poly,
        "Row": {
            "UVectECF": u_row,
            "SS": row_ss,
            "ImpRespWid": row_imp_res_wid,
            "Sgn": row_sgn,
            "ImpRespBW": row_imp_res_bw,
            "KCtr": row_kctr,
            "DeltaK1": -row_imp_res_bw / 2,
            "DeltaK2": row_imp_res_bw / 2,
            "DeltaKCOAPoly": row_deltak_coa_poly,
            "WgtType": {
                "WindowName": row_win_name,
                "Parameter": [("COEFFICIENT", str(row_win_shape))],
            },
            "WgtFunct": row_wgts,
        },
        "Col": {
            "UVectECF": u_col,
            "SS": col_ss,
            "ImpRespWid": col_imp_res_wid,
            "Sgn": col_sgn,
            "ImpRespBW": col_imp_res_bw,
            "KCtr": col_kctr,
            "DeltaK1": dk1,
            "DeltaK2": dk2,
            "DeltaKCOAPoly": col_deltak_coa_poly,
            "WgtFunct": col_wgts,
        },
    }

    sicd_ew["Timeline"] = {
        "CollectStart": collect_start,
        "CollectDuration": collect_duration,
        "IPP": {
            "@size": 1,
            "Set": [
                {
                    "@index": 1,
                    "TStart": t_start,
                    "TEnd": t_end,
                    "IPPStart": ipp_start,
                    "IPPEnd": ipp_end,
                    "IPPPoly": ipp_poly,
                }
            ],
        },
    }

    sicd_ew["Position"]["ARPPoly"] = apc_poly

    sicd_ew["RadarCollection"] = {
        "TxFrequency": {
            "Min": acq_tx_freq_min,
            "Max": acq_tx_freq_max,
        },
        "Waveform": {
            "@size": 1,
            "WFParameters": [
                {
                    "@index": 1,
                    "TxRFBandwidth": acq_rf_bw,
                }
            ],
        },
        "TxPolarization": tx_polarization,
        "RcvChannels": {
            "@size": 1,
            "ChanParameters": [
                {
                    "@index": 1,
                    "TxRcvPolarization": tx_rcv_polarization,
                }
            ],
        },
    }

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    sicd_ew["ImageFormation"] = {
        "RcvChanProc": {
            "NumChanProc": 1,
            "ChanIndex": [1],
        },
        "TxRcvPolarizationProc": tx_rcv_polarization_proc,
        "TStartProc": t_start_proc,
        "TEndProc": t_end_proc,
        "TxFrequencyProc": {
            "MinProc": proc_freq_min,
            "MaxProc": proc_freq_max,
        },
        "ImageFormAlgo": "RMA",
        "STBeamComp": "NO",
        "ImageBeamComp": "NO",
        "AzAutofocus": "NO",
        "RgAutofocus": "NO",
        "Processing": [
            {
                "Type": f"sarkit-convert {__version__} @ {now}",
                "Applied": True,
            },
        ],
    }

    sicd_ew["RMA"] = {
        "RMAlgoType": "OMEGA_K",
        "ImageType": "INCA",
        "INCA": {
            "TimeCAPoly": time_ca_poly,
            "R_CA_SCP": scp_rg,
            "FreqZero": proc_center_frequency,
            "DRateSFPoly": drsf_poly,
            "DopCentroidPoly": doppler_centroid_poly,
        },
    }

    sicd_ew["SCPCOA"] = sksicd.compute_scp_coa(sicd_xml_obj.getroottree())

    # Update ImageCorners
    sicd_xmltree = sicd_xml_obj.getroottree()
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
    sicd_ew["GeoData"]["ImageCorners"] = icp_llh[:, :2]

    # Check for XML consistency
    sicd_con = sarkit.verification.SicdConsistency(sicd_xmltree)
    sicd_con.check()
    sicd_con.print_result(fail_detail=True)

    # Grab the data
    with h5py.File(h5_filename, "r") as h5file:
        datapath = f"science/LSAR/RSLC/swaths/{freq_str}/{polarization}"
        data_arr = np.asarray(h5file[datapath])
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
            "ostaid": creation_site,
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


def discover_images(h5_filename):
    def as_str(item):
        return item[...].astype(str).tolist()

    images = list()
    with h5py.File(h5_filename, "r") as h5file:
        frequencies = as_str(h5file["/science/LSAR/identification/listOfFrequencies"])
        for freq in frequencies:
            path = "/science/LSAR/RSLC/swaths/frequency" + freq + "/listOfPolarizations"
            polarizations = as_str(h5file[path])
            for pol in polarizations:
                images.append((freq, pol))

    return images


def main(args=None):
    """CLI for converting NISAR RSLC to SICD"""
    parser = argparse.ArgumentParser(
        description="Converts a NISAR HDF5 product into a SICD.",
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
        help="path of the output SICD file.  The strings '{freq}' and '{pol}' will be replaced as appropriate for multiple images",
    )
    config = parser.parse_args(args)

    images = discover_images(config.input_h5_file)

    for frequency, polarization in images:
        fname = config.output_sicd_file.name.format(freq=frequency, pol=polarization)
        output_sicd = config.output_sicd_file.with_name(fname)
        hdf5_to_sicd(
            config.input_h5_file,
            output_sicd,
            frequency=frequency,
            polarization=polarization,
            classification=config.classification,
        )


if __name__ == "__main__":
    main()
