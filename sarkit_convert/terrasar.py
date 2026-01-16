"""
===================
TSX Complex to SICD
===================

Convert a complex image from the TerraSAR-X COSAR into SICD 1.4

During development, the following documents were considered:

"Level 1b Product Format Specification", Issue 1.3

In addition, SARPy was consulted on how to use the TSX metadata to compute SICD
metadata that would predict the complex data characteristics

"""

import argparse
import datetime
import pathlib

import dateutil.parser
import lxml.etree
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import sarkit.verification
import sarkit.wgs84
import scipy.constants

from sarkit_convert import __version__
from sarkit_convert import _utils as utils

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}

COSAR_PIXEL_TYPE = "RE16I_IM16I"

MODE_TYPE_MAP = {
    "ST": "SPOTLIGHT",
    "SL": "DYNAMIC STRIPMAP",
    "HS": "DYNAMIC STRIPMAP",
    "SM": "STRIPMAP",
}

INSTRUMENT_CONVERSION_TO_SECONDS = 32 / 3.29658384e8
INSTRUMENT_CONVERSION_TO_HZ = 1.25e6


def _get_xyz(root, prefix):
    return [
        float(root.findtext(f"{prefix}{component}")) for component in ("X", "Y", "Z")
    ]


def _parse_to_naive(timestamp_str):
    return dateutil.parser.parse(timestamp_str).replace(tzinfo=None)


def _boolstr_to_bool(text):
    return text in ("1", "true")


def _compute_apc_poly(tsx_xml, start_time, stop_time):
    """Creates an Aperture Phase Center (APC) poly that best fits the provided state vectors

    Polynomial generates 3D coords in ECF as a function of time from start of collect.

    Parameters
    ----------
    tsx_xml: lxml.Element
        The TSX xml
    start_time: datetime.datetime
        The start time to fit
    stop_time: datetime.datetime
        The end time to fit

    Returns
    -------
    `numpy.ndarray`, shape=(6, 3)
        APC poly
    """

    times = []
    positions = []
    velocities = []
    for state_vec in tsx_xml.findall("./platform/orbit/stateVec"):
        sv_time = _parse_to_naive(state_vec.findtext("./timeUTC"))
        times.append((sv_time - start_time).total_seconds())
        positions.append(_get_xyz(state_vec, "pos"))
        velocities.append(_get_xyz(state_vec, "vel"))

    start = 0.0
    stop = (stop_time - start_time).total_seconds()
    apc_poly = utils.fit_state_vectors(
        (start, stop),
        times,
        positions,
        velocities,
        order=5,
    )

    return apc_poly


def read_cosar(cosar_file):
    burst_annot_dtype = np.dtype(
        [
            ("BIB", ">i4"),
            ("RSRI", ">i4"),
            ("RS", ">i4"),
            ("AS", ">i4"),
            ("BI", ">i4"),
            ("RTNB", ">i4"),
            ("TNL", ">i4"),
        ]
    )
    burst_annot = np.fromfile(cosar_file, dtype=burst_annot_dtype, count=1)
    assert (
        burst_annot["BIB"] == burst_annot["RTNB"] * burst_annot["TNL"]
    )  # TODO Handle only one burst for now
    rtnb = burst_annot["RTNB"][0]
    n_rg = burst_annot["RS"][0]
    n_az = burst_annot["AS"][0]
    azim_annot_shape = (3, n_rg + 2)
    azim_info = np.fromfile(
        cosar_file, dtype=">i4", count=np.prod(azim_annot_shape), offset=rtnb
    ).reshape(azim_annot_shape)[:, 2:]
    if not np.array_equiv(azim_info[0, 2:], 0):
        raise ValueError("COSAR image data is not deskewed")

    range_lines_shape = (n_az, n_rg + 2)
    range_lines = np.fromfile(
        cosar_file, dtype=">i4", count=np.prod(range_lines_shape), offset=4 * rtnb
    ).reshape(range_lines_shape)

    range_info = range_lines[:, :2]
    assert np.all(range_info[:, 0] <= range_info[:, 1])

    view_dtype = sksicd.PIXEL_TYPES[COSAR_PIXEL_TYPE]["dtype"].newbyteorder("big")
    data_arr = range_lines[:, 2:]
    complex_data_arr = np.squeeze(data_arr.view(view_dtype))

    return complex_data_arr


def cosar_to_sicd(
    tsx_xml,
    layer_index,
    cosar_file,
    sicd_file,
    classification,
    chan_index,
    tx_polarizations,
    tx_rcv_pols,
):
    cal_const_elem = tsx_xml.find(
        f'./calibration/calibrationConstant[@layerIndex="{layer_index}"]'
    )
    pol_layer = cal_const_elem.findtext("./polLayer")
    settings_elem = tsx_xml.xpath(
        f'./instrument/settings/polLayer[text()="{pol_layer}"]/..'
    )[0]

    # Timeline
    proc_param = tsx_xml.find("./processing/processingParameter")
    collection_start_time = _parse_to_naive(
        proc_param.findtext("./rangeCompression/segmentInfo/dataSegment/startTimeUTC")
    )
    collection_stop_time = _parse_to_naive(
        proc_param.findtext("./rangeCompression/segmentInfo/dataSegment/stopTimeUTC")
    )
    collection_duration = (collection_stop_time - collection_start_time).total_seconds()
    prf = float(settings_elem.findtext("./settingRecord/PRF"))
    num_pulses = int(np.ceil(collection_duration * prf))
    look = {"LEFT": 1, "RIGHT": -1}[
        tsx_xml.findtext("./productInfo/acquisitionInfo/lookDirection")
    ]

    # Collection Info
    collector_name = tsx_xml.findtext("./productInfo/missionInfo/mission")
    core_name = tsx_xml.findtext("./productInfo/sceneInfo/sceneID")
    radar_mode_id = tsx_xml.findtext("./productInfo/acquisitionInfo/imagingMode")
    if radar_mode_id not in MODE_TYPE_MAP:
        raise ValueError(f"Unsupported mode {radar_mode_id}")
    radar_mode_type = MODE_TYPE_MAP[radar_mode_id]

    # Creation Info
    creation_time = _parse_to_naive(tsx_xml.findtext("./generalHeader/generationTime"))
    generation_system = tsx_xml.find("./generalHeader/generationSystem")
    application = generation_system.text
    version = generation_system.attrib["version"]
    creation_application = f"{application} version {version}"
    originator_facility = tsx_xml.findtext(
        "./productInfo/generationInfo/level1ProcessingFacility"
    )

    # COSAR data has range in the fast dimension, so this is a deliberate transposition
    # Image Data
    num_rows = int(
        tsx_xml.findtext("./productInfo/imageDataInfo/imageRaster/numberOfColumns")
    )
    num_cols = int(
        tsx_xml.findtext("./productInfo/imageDataInfo/imageRaster/numberOfRows")
    )
    first_row = 0
    first_col = 0
    scp_row = int(
        tsx_xml.findtext("./productInfo/sceneInfo/sceneCenterCoord/refColumn")
    )
    ref_col = int(tsx_xml.findtext("./productInfo/sceneInfo/sceneCenterCoord/refRow"))
    if look > 0:
        scp_col = num_cols - 1 - ref_col
    else:
        scp_col = ref_col
    scp_pixel = np.array([scp_row, scp_col])

    # Position
    apc_poly = _compute_apc_poly(tsx_xml, collection_start_time, collection_stop_time)

    # Image Formation
    st_beam_comp = "NONE"
    procflags_elem = tsx_xml.find("./processing/processingFlags")
    if _boolstr_to_bool(procflags_elem.findtext("./azimuthPatternCorrectedFlag")):
        st_beam_comp = "GLOBAL"
    if _boolstr_to_bool(
        procflags_elem.findtext("./spotLightBeamCorrectedFlag")
    ) or _boolstr_to_bool(procflags_elem.findtext("./scanSARBeamCorrectedFlag")):
        st_beam_comp = "SV"

    # Radar Collection
    center_frequency = float(
        tsx_xml.findtext("./instrument/radarParameters/centerFrequency")
    )
    chirp_elem = proc_param.find("./rangeCompression/chirps/referenceChirp")
    tx_pulse_length = (
        float(chirp_elem.findtext("./pulseLength")) * INSTRUMENT_CONVERSION_TO_SECONDS
    )
    tx_rf_bw = (
        float(chirp_elem.findtext("./pulseBandwidth")) * INSTRUMENT_CONVERSION_TO_HZ
    )
    tx_fm_rate = (
        {"UP": 1, "DOWN": -1}[chirp_elem.findtext("./chirpSlope")]
        * tx_rf_bw
        / tx_pulse_length
    )
    tx_freq_min = center_frequency - 0.5 * tx_rf_bw
    tx_freq_max = center_frequency + 0.5 * tx_rf_bw
    tx_freq_start = center_frequency - (tx_pulse_length / 2 * tx_fm_rate)
    adc_sample_rate = float(settings_elem.findtext("./RSF"))
    rcv_window_length = (
        float(settings_elem.findtext("./settingRecord/echowindowLength"))
        / adc_sample_rate
    )
    tx_rcv_polarization = tx_rcv_pols[chan_index - 1]
    tx_polarization = tx_rcv_polarization[0]

    # Grid
    orientation = tsx_xml.findtext(
        "./productSpecific/complexImageInfo/imageDataStartWith"
    )
    if orientation != "EARLYAZNEARRG":
        raise ValueError("Image data not in expected orientation")
    # These spacings are within a row/col, so effectively cancel the raster transpose.
    intervals = np.array(
        [
            float(
                tsx_xml.findtext("./productInfo/imageDataInfo/imageRaster/rowSpacing")
            ),
            float(
                tsx_xml.findtext(
                    "./productInfo/imageDataInfo/imageRaster/columnSpacing"
                )
            ),
        ]
    )
    spacings = np.array(
        [
            intervals[0] * scipy.constants.speed_of_light / 2,
            float(
                tsx_xml.findtext(
                    "./productSpecific/complexImageInfo/projectedSpacingAzimuth"
                )
            ),
        ]
    )
    row_bw = (
        float(proc_param.findtext("./rangeLookBandwidth"))
        * 2
        / scipy.constants.speed_of_light
    )
    col_bw = (
        float(proc_param.findtext("./azimuthLookBandwidth"))
        * intervals[1]
        / spacings[1]
    )
    zd_0_utc = _parse_to_naive(
        tsx_xml.findtext("./productInfo/sceneInfo/start/timeUTC")[:-1]
    )
    zd_az_0 = (zd_0_utc - collection_start_time).total_seconds()
    zd_rg_0 = float(tsx_xml.findtext("./productInfo/sceneInfo/rangeTime/firstPixel"))
    row_wid = 1 / row_bw
    col_wid = 1 / col_bw

    # Create doppler rate and centroid
    num_grid_pts = 51
    range_times = np.linspace(0, num_rows - 1, num_grid_pts) * intervals[0] + zd_rg_0
    xrow_vals = ((range_times - zd_rg_0) / intervals[0] - scp_pixel[0]) * spacings[0]

    def get_poly_vals(base_path, rel_poly_path):
        values = []
        azimuth_times = []
        for base_node in tsx_xml.findall(base_path):
            utc_time = _parse_to_naive(base_node.findtext("./timeUTC"))
            azimuth_times.append((utc_time - collection_start_time).total_seconds())
            poly_node = base_node.find(rel_poly_path)
            ref_range_time = float(poly_node.findtext("referencePoint"))
            degree = int(poly_node.findtext("polynomialDegree"))
            poly = np.zeros(degree + 1)
            for coef in poly_node.findall("coefficient"):
                poly[int(coef.attrib["exponent"])] = float(coef.text)
            current_values = npp.polyval(range_times - ref_range_time, poly)
            values.append(current_values)
        values = np.stack(values, axis=1)
        return np.array(azimuth_times), values

    def zd_times_to_ycol(times):
        indices = (np.asarray(times) - zd_az_0) / intervals[1]
        if look > 0:
            ycol_vals = (num_cols - 1 - indices - scp_pixel[1]) * spacings[1]
        else:
            ycol_vals = (indices - scp_pixel[1]) * spacings[1]
        return ycol_vals

    drate_times, doppler_rate = get_poly_vals(
        "./processing/geometry/dopplerRate", "dopplerRatePolynomial"
    )
    drate_ycol_vals = zd_times_to_ycol(drate_times)
    drate_grid_coords = np.stack(
        np.meshgrid(xrow_vals, drate_ycol_vals, indexing="ij"), axis=-1
    )
    doppler_rate_poly = utils.polyfit2d_tol(
        drate_grid_coords[..., 0].flatten(),
        drate_grid_coords[..., 1].flatten(),
        doppler_rate.flatten(),
        4,
        4,
        1e-3,
    )

    centroid_coord_type = tsx_xml.findtext(
        "./processing/doppler/dopplerCentroidCoordinateType"
    )
    dopp_cent_path = (
        f'./processing/doppler/dopplerCentroid[@layerIndex="{layer_index}"]'
    )
    if radar_mode_type == "SPOTLIGHT":
        coa_time = collection_duration / 2
        delta_time = coa_time - np.array(drate_times)
        doppler_centroid = (
            npp.polyval2d(
                drate_grid_coords[..., 0], drate_grid_coords[..., 1], doppler_rate_poly
            )
            * delta_time[np.newaxis, :]
        )
        doppler_centroid_poly = utils.polyfit2d_tol(
            drate_grid_coords[..., 0].flatten(),
            drate_grid_coords[..., 1].flatten(),
            doppler_centroid.flatten(),
            4,
            4,
            1e-2,
        )
        time_coa_poly = np.array([[coa_time]])
    else:
        if centroid_coord_type == "RAW":
            coa_times, doppler_centroid = get_poly_vals(
                dopp_cent_path + "/dopplerEstimate", "./combinedDoppler"
            )
            coa_times = np.tile(coa_times, (doppler_centroid.shape[0], 1))
            zd_times = coa_times - doppler_centroid / doppler_rate_poly[0, 0]
            dcent_ycol_vals = zd_times_to_ycol(zd_times)
            dcent_xrow_vals = np.tile(
                xrow_vals[:, np.newaxis], (1, dcent_ycol_vals.shape[1])
            )
            dcent_grid_coords = np.stack((dcent_xrow_vals, dcent_ycol_vals), axis=-1)
            doppler_centroid_poly = utils.polyfit2d_tol(
                dcent_grid_coords[..., 0].flatten(),
                dcent_grid_coords[..., 1].flatten(),
                doppler_centroid.flatten(),
                4,
                4,
                1e-2,
            )
        elif centroid_coord_type == "ZERODOPPLER":
            zd_times, doppler_centroid = get_poly_vals(
                dopp_cent_path + "/dopplerEstimate", "./combinedDoppler"
            )
            zd_times = np.tile(zd_times, (doppler_centroid.shape[0], 1))
            coa_times = zd_times + doppler_centroid / doppler_rate_poly[0, 0]
            dcent_ycol_vals = zd_times_to_ycol(zd_times)
            dcent_xrow_vals = np.tile(
                xrow_vals[:, np.newaxis], (1, dcent_ycol_vals.shape[1])
            )
            dcent_grid_coords = np.stack((dcent_xrow_vals, dcent_ycol_vals), axis=-1)
            doppler_centroid_poly = utils.polyfit2d_tol(
                dcent_grid_coords[..., 0].flatten(),
                dcent_grid_coords[..., 1].flatten(),
                doppler_centroid.flatten(),
                4,
                4,
                1e-2,
            )
        else:
            raise ValueError(
                f"Unknown dopplerCentroidCoordinateType: {centroid_coord_type}"
            )
        time_coa_poly = utils.polyfit2d_tol(
            dcent_grid_coords[..., 0].flatten(),
            dcent_grid_coords[..., 1].flatten(),
            coa_times.flatten(),
            4,
            4,
            1e-3,
        )

    range_rate_per_hz = -scipy.constants.speed_of_light / (2 * center_frequency)
    range_rate_rate = doppler_rate * range_rate_per_hz

    if look > 0:
        scp_tca = (num_cols - 1 - scp_col) * intervals[1] + zd_az_0
    else:
        scp_tca = scp_col * intervals[1] + zd_az_0
    scp_rca = (
        (zd_rg_0 + scp_pixel[0] * intervals[0]) * scipy.constants.speed_of_light / 2
    )
    time_ca_poly = np.array([scp_tca, -look * intervals[1] / spacings[1]])
    range_ca = range_times * scipy.constants.speed_of_light / 2
    speed_ca = npl.norm(npp.polyval(drate_times, npp.polyder(apc_poly)), axis=0)
    drsf = range_rate_rate * range_ca[:, np.newaxis] / speed_ca[np.newaxis, :] ** 2
    drsf_poly = utils.polyfit2d_tol(
        drate_grid_coords[..., 0].flatten(),
        drate_grid_coords[..., 1].flatten(),
        drsf.flatten(),
        4,
        4,
        1e-6,
    )

    si_elem = tsx_xml.find("./productInfo/sceneInfo")
    scc_llh = np.array(
        [
            float(si_elem.findtext("./sceneCenterCoord/lat")),
            float(si_elem.findtext("./sceneCenterCoord/lon")),
            float(si_elem.findtext("./sceneAverageHeight")),
        ]
    )
    scc_ecf = sarkit.wgs84.geodetic_to_cartesian(scc_llh)

    scp_drsf = drsf_poly[0, 0]
    scp_tcoa = time_coa_poly[0, 0]
    scp_delta_t_coa = scp_tcoa - scp_tca
    scp_varp_ca_mag = npl.norm(npp.polyval(scp_tca, npp.polyder(apc_poly)))
    scp_rcoa = np.sqrt(scp_rca**2 + scp_drsf * scp_varp_ca_mag**2 * scp_delta_t_coa**2)
    scp_rratecoa = scp_drsf / scp_rcoa * scp_varp_ca_mag**2 * scp_delta_t_coa

    scp_set = sksicd.projection.ProjectionSetsMono(
        t_COA=np.array([scp_tcoa]),
        ARP_COA=np.array([npp.polyval(scp_tcoa, apc_poly)]),
        VARP_COA=np.array([npp.polyval(scp_tcoa, npp.polyder(apc_poly))]),
        R_COA=np.array([scp_rcoa]),
        Rdot_COA=np.array([scp_rratecoa]),
    )
    scp_ecf, _, _ = sksicd.projection.r_rdot_to_constant_hae_surface(
        look, scc_ecf, scp_set, scc_llh[2]
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

    # Antenna
    attitude_elem = tsx_xml.find("./platform/attitude")
    attitude_utcs = []
    attitude_quaternions = []
    for attitude_data in attitude_elem.findall("./attitudeData"):
        attitude_utcs.append(_parse_to_naive(attitude_data.findtext("./timeUTC")))
        quat = [
            float(attitude_data.findtext("./q0")),
            float(attitude_data.findtext("./q1")),
            float(attitude_data.findtext("./q2")),
            float(attitude_data.findtext("./q3")),
        ]
        attitude_quaternions.append(quat)
    attitude_quaternions = np.array(attitude_quaternions)

    def get_mech_frame_from_quat(att_quat):
        scipy_quat = np.roll(att_quat, -1)
        return scipy.spatial.transform.Rotation.from_quat(scipy_quat).inv().as_matrix()

    rel_att_times = np.array(
        [(att_utc - collection_start_time).total_seconds() for att_utc in attitude_utcs]
    )
    good_indices = np.where(
        np.logical_and(
            np.less(-60, rel_att_times),
            np.less(rel_att_times, 60 + collection_duration),
        )
    )
    good_times = rel_att_times[good_indices]
    good_att_quat = attitude_quaternions[good_indices]
    mech_frame = np.array(
        [get_mech_frame_from_quat(att_quat) for att_quat in good_att_quat]
    )

    ux_rot = mech_frame[:, 0, :]
    uy_rot = mech_frame[:, 1, :]

    ant_x_dir_poly = utils.fit_state_vectors(
        (0, (collection_stop_time - collection_start_time).total_seconds()),
        good_times,
        ux_rot,
        None,
        None,
        order=4,
    )
    ant_y_dir_poly = utils.fit_state_vectors(
        (0, (collection_stop_time - collection_start_time).total_seconds()),
        good_times,
        uy_rot,
        None,
        None,
        order=4,
    )

    freq_zero = center_frequency
    antenna_pattern_elem = tsx_xml.xpath(
        f'./calibration/calibrationData/antennaPattern[polLayer[text()="{pol_layer}"]]'
    )[0]
    antenna_beam_elevation = np.arctan2(
        float(antenna_pattern_elem.findtext("./beamPointingVector/y")),
        float(antenna_pattern_elem.findtext("./beamPointingVector/z")),
    )
    spotlight_elem = tsx_xml.find(
        "./productInfo/acquisitionInfo/imagingModeSpecificInfo/spotLight"
    )
    if spotlight_elem is not None:
        azimuth_steering = np.array(
            [
                float(spotlight_elem.findtext("./azimuthSteeringAngleFirst")),
                float(spotlight_elem.findtext("./azimuthSteeringAngleLast")),
            ]
        )
        eb_dcx_poly = npp.polyfit(
            [0, collection_duration], np.sin(np.deg2rad(azimuth_steering)), 1
        )
    else:
        eb_dcx_poly = [0.0]
    eb_dcy_poly = [-look * np.sin(antenna_beam_elevation)]

    def get_angles_gains(pattern_elem):
        antenna_angles = []
        antenna_gains = []
        for gain_ext in pattern_elem.findall("./gainExt"):
            antenna_angles.append(float(gain_ext.attrib["angle"]))
            antenna_gains.append(float(gain_ext.text))
        return np.array(antenna_angles), np.array(antenna_gains)

    antenna_rg_angles, antenna_rg_gains = get_angles_gains(
        antenna_pattern_elem.find("./elevationPattern")
    )
    antenna_az_angles, antenna_az_gains = get_angles_gains(
        antenna_pattern_elem.find("./azimuthPattern")
    )

    fit_order = 4

    def fit_gains(angles, gains):
        fit_limit = -9
        array_mask = gains > fit_limit
        dcs = np.sin(np.deg2rad(angles))
        return npp.polyfit(dcs[array_mask], gains[array_mask], fit_order)

    antenna_array_gain = np.zeros((fit_order + 1, fit_order + 1), dtype=float)
    antenna_array_gain[0, :] = fit_gains(
        antenna_rg_angles + look * np.rad2deg(antenna_beam_elevation),
        antenna_rg_gains,
    )
    antenna_array_gain[:, 0] = fit_gains(antenna_az_angles, antenna_az_gains)
    antenna_array_gain[0, 0] = 0.0

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
        "DateTime": creation_time,
    }
    sicd_ew["ImageData"] = {
        "PixelType": COSAR_PIXEL_TYPE,
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

    row_window_name = proc_param.findtext("./rangeWindowID")
    row_window_coeff = float(proc_param.findtext("./rangeWindowCoefficient"))
    col_window_name = proc_param.findtext("./azimuthWindowID")
    col_window_coeff = float(proc_param.findtext("./azimuthWindowCoefficient"))

    sicd_ew["Grid"] = {
        "ImagePlane": "SLANT",
        "Type": "RGZERO",
        "TimeCOAPoly": time_coa_poly,
        "Row": {
            "UVectECF": u_row,
            "SS": spacings[0],
            "ImpRespWid": row_wid,
            "Sgn": -1,
            "ImpRespBW": row_bw,
            "KCtr": center_frequency / (scipy.constants.speed_of_light / 2),
            "DeltaK1": -row_bw / 2,
            "DeltaK2": row_bw / 2,
            "DeltaKCOAPoly": [[0.0]],
            "WgtType": {
                "WindowName": row_window_name,
                "Parameter": [("COEFFICIENT", str(row_window_coeff))],
            },
        },
        "Col": {
            "UVectECF": u_col,
            "SS": spacings[1],
            "ImpRespWid": col_wid,
            "Sgn": -1,
            "ImpRespBW": col_bw,
            "KCtr": 0.0,
            "DeltaK1": dk1,
            "DeltaK2": dk2,
            "DeltaKCOAPoly": col_deltakcoa_poly,
            "WgtType": {
                "WindowName": col_window_name,
                "Parameter": [("COEFFICIENT", str(col_window_coeff))],
            },
        },
    }
    rcs_row_sf = None
    rcs_col_sf = None
    if row_window_name == "Hamming":
        wgts = scipy.signal.windows.general_hamming(512, row_window_coeff, sym=True)
        sicd_ew["Grid"]["Row"]["WgtFunct"] = wgts
        row_broadening_factor = utils.broadening_from_amp(wgts)
        row_wid = row_broadening_factor / row_bw
        sicd_ew["Grid"]["Row"]["ImpRespWid"] = row_wid
        rcs_row_sf = 1 + np.var(wgts) / np.mean(wgts) ** 2
    if col_window_name == "Hamming":
        wgts = scipy.signal.windows.general_hamming(512, col_window_coeff, sym=True)
        sicd_ew["Grid"]["Col"]["WgtFunct"] = wgts
        col_broadening_factor = utils.broadening_from_amp(wgts)
        col_wid = col_broadening_factor / col_bw
        sicd_ew["Grid"]["Col"]["ImpRespWid"] = col_wid
        rcs_col_sf = 1 + np.var(wgts) / np.mean(wgts) ** 2

    sicd_ew["Timeline"] = {
        "CollectStart": collection_start_time,
        "CollectDuration": collection_duration,
        "IPP": {
            "@size": 1,
            "Set": [
                {
                    "@index": 1,
                    "TStart": 0,
                    "TEnd": num_pulses / prf,
                    "IPPStart": 0,
                    "IPPEnd": num_pulses - 1,
                    "IPPPoly": [0, prf],
                }
            ],
        },
    }

    sicd_ew["Position"]["ARPPoly"] = apc_poly

    chan_parameters = []
    for ndx, tx_rcv_pol in enumerate(tx_rcv_pols):
        chan_parameters.append(
            {
                "@index": ndx + 1,
                "TxRcvPolarization": tx_rcv_pol,
            }
        )

    sicd_ew["RadarCollection"] = {
        "TxFrequency": {
            "Min": tx_freq_min,
            "Max": tx_freq_max,
        },
        "Waveform": {
            "@size": 1,
            "WFParameters": [
                {
                    "@index": 1,
                    "TxPulseLength": tx_pulse_length,
                    "TxRFBandwidth": tx_rf_bw,
                    "TxFreqStart": tx_freq_start,
                    "TxFMRate": tx_fm_rate,
                    "RcvWindowLength": rcv_window_length,
                    "ADCSampleRate": adc_sample_rate,
                }
            ],
        },
        "TxPolarization": tx_polarization,
        "RcvChannels": {
            "@size": len(tx_rcv_pols),
            "ChanParameters": chan_parameters,
        },
    }
    if len(tx_polarizations) > 1:
        sicd_ew["RadarCollection"]["TxPolarization"] = "SEQUENCE"
        tx_steps = []
        for ndx, tx_pol in enumerate(tx_polarizations):
            tx_steps.append(
                {
                    "@index": ndx + 1,
                    "TxPolarization": tx_pol,
                }
            )
        sicd_ew["RadarCollection"]["TxSequence"] = {
            "@size": len(tx_polarizations),
            "TxStep": tx_steps,
        }

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    sicd_ew["ImageFormation"] = {
        "RcvChanProc": {
            "NumChanProc": 1,
            "ChanIndex": [chan_index],
        },
        "TxRcvPolarizationProc": tx_rcv_polarization,
        "TStartProc": 0,
        "TEndProc": collection_duration,
        "TxFrequencyProc": {
            "MinProc": tx_freq_min,
            "MaxProc": tx_freq_max,
        },
        "ImageFormAlgo": "RMA",
        "STBeamComp": st_beam_comp,
        "ImageBeamComp": "SV",
        "AzAutofocus": "NO",
        "RgAutofocus": "NO",
        "Processing": [
            {
                "Type": f"sarkit-convert {__version__} @ {now}",
                "Applied": True,
            },
        ],
    }

    sicd_ew["Antenna"]["TwoWay"]["XAxisPoly"] = ant_x_dir_poly
    sicd_ew["Antenna"]["TwoWay"]["YAxisPoly"] = ant_y_dir_poly
    sicd_ew["Antenna"]["TwoWay"]["FreqZero"] = freq_zero
    sicd_ew["Antenna"]["TwoWay"]["EB"]["DCXPoly"] = eb_dcx_poly
    sicd_ew["Antenna"]["TwoWay"]["EB"]["DCYPoly"] = eb_dcy_poly
    sicd_ew["Antenna"]["TwoWay"]["Array"]["GainPoly"] = antenna_array_gain
    sicd_ew["Antenna"]["TwoWay"]["Array"]["PhasePoly"] = np.zeros(
        dtype=float, shape=(1, 1)
    )

    sicd_ew["RMA"] = {
        "RMAlgoType": "OMEGA_K",
        "ImageType": "INCA",
        "INCA": {
            "TimeCAPoly": time_ca_poly,
            "R_CA_SCP": scp_rca,
            "FreqZero": center_frequency,
            "DRateSFPoly": drsf_poly,
            "DopCentroidPoly": doppler_centroid_poly,
        },
    }

    sicd_ew["SCPCOA"] = sksicd.compute_scp_coa(sicd_xml_obj.getroottree())

    # Add Radiometric
    cal_constant = float(cal_const_elem.findtext("./calFactor"))
    betazero_poly = np.array([[cal_constant]])
    graze = np.deg2rad(float(sicd_xml_obj.findtext("./{*}SCPCOA/{*}GrazeAng")))
    twist = np.deg2rad(float(sicd_xml_obj.findtext("./{*}SCPCOA/{*}TwistAng")))
    sigmazero_poly = betazero_poly * np.cos(graze) * np.cos(twist)
    gammazero_poly = betazero_poly / np.tan(graze) * np.cos(twist)

    noise_path = f'./noise[@layerIndex="{layer_index}"]'
    noise_times, noise_vals = get_poly_vals(f"{noise_path}/imageNoise", "noiseEstimate")
    noise_ycol_vals = zd_times_to_ycol(np.tile(noise_times, (noise_vals.shape[0], 1)))
    noise_xrow_vals = np.tile(xrow_vals[:, np.newaxis], (1, noise_ycol_vals.shape[1]))
    noise_grid_coords = np.stack((noise_xrow_vals, noise_ycol_vals), axis=-1)
    fit_noise_vals = 10 * np.log10(noise_vals)
    noise_poly = utils.polyfit2d_tol(
        noise_grid_coords[..., 0].flatten(),
        noise_grid_coords[..., 1].flatten(),
        fit_noise_vals.flatten(),
        min(fit_noise_vals.shape[0] - 1, 4),
        min(fit_noise_vals.shape[1] - 1, 4),
        1e-2,
    )

    sicd_ew["Radiometric"] = {
        "NoiseLevel": {
            "NoiseLevelType": "ABSOLUTE",
            "NoisePoly": noise_poly,
        },
        "SigmaZeroSFPoly": sigmazero_poly,
        "BetaZeroSFPoly": betazero_poly,
        "GammaZeroSFPoly": gammazero_poly,
    }

    if rcs_row_sf and rcs_col_sf:
        rcssf_poly = betazero_poly * (rcs_row_sf * rcs_col_sf / (row_bw * col_bw))
        sicd_ew["Radiometric"]["RCSSFPoly"] = rcssf_poly

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
    sicd_ew["GeoData"]["ImageCorners"] = icp_llh[:, :2]

    # Add RNIIRS
    inf_density, pred_rniirs = utils.get_rniirs_estimate(sicd_ew)
    sicd_ew["CollectionInfo"].add(
        "Parameter", ("INFORMATION_DENSITY", f"{inf_density:.2g}")
    )
    sicd_ew["CollectionInfo"].add(
        "Parameter", ("PREDICTED_RNIIRS", f"{pred_rniirs:.1f}")
    )

    # Validate XML
    sicd_con = sarkit.verification.SicdConsistency(sicd_xmltree)
    sicd_con.check()
    sicd_con.print_result(fail_detail=True)

    # Grab the data
    complex_data_arr = np.transpose(read_cosar(cosar_file))
    if look > 0:
        complex_data_arr = complex_data_arr[:, ::-1]

    metadata = sksicd.NitfMetadata(
        xmltree=sicd_xmltree,
        file_header_part={
            "ostaid": originator_facility,
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

    with sicd_file.open("wb") as f:
        with sksicd.NitfWriter(f, metadata) as writer:
            writer.write_image(complex_data_arr)


def main(args=None):
    parser = argparse.ArgumentParser(description="Converts a TSX dataset into a SICD.")

    parser.add_argument(
        "input_xml_file", type=pathlib.Path, help="path of the input XML file"
    )
    parser.add_argument(
        "classification",
        help="content of the /SICD/CollectionInfo/Classification node in the SICD XML",
    )
    parser.add_argument(
        "output_sicd_file",
        type=pathlib.Path,
        help='path of the output SICD file. The string "{pol}" will be replaced with polarization for multiple images',
    )
    config = parser.parse_args(args)

    if not config.input_xml_file.is_file():
        raise ValueError(f"Input XML file {str(config.input_xml_file)} is not a file")

    tsx_xml = lxml.etree.parse(config.input_xml_file).getroot()

    images = dict()
    img_ndx = 1
    tx_polarizations = []
    tx_rcv_pols = []
    for imagedata_elem in tsx_xml.findall("./productComponents/imageData"):
        layer_index = imagedata_elem.attrib["layerIndex"]
        pol_layer = imagedata_elem.findtext("./polLayer")
        assert imagedata_elem.findtext("./file/location/host") == "."
        path = imagedata_elem.findtext("./file/location/path")
        fname = imagedata_elem.findtext("./file/location/filename")
        cosar_filename = config.input_xml_file.parent / path / fname
        if not cosar_filename.is_file():
            raise ValueError(f"Input COSAR file {str(cosar_filename)} is not a file")

        sicd_filename = pathlib.Path(str(config.output_sicd_file).format(pol=pol_layer))
        images[layer_index] = {
            "pol_layer": pol_layer,
            "chan_index": img_ndx,
            "cosar_filename": cosar_filename,
            "sicd_filename": sicd_filename,
        }
        tx_rcv_pols.append(f"{pol_layer[0]}:{pol_layer[1]}")
        if (tx_polarization := pol_layer[0]) not in tx_polarizations:
            tx_polarizations.append(tx_polarization)
        img_ndx += 1

    if len(images) != len(set([image["sicd_filename"] for image in images.values()])):
        raise ValueError("Output filename does not include necessary polarization slug")

    for layer_index, img_info in images.items():
        cosar_to_sicd(
            tsx_xml=tsx_xml,
            layer_index=layer_index,
            cosar_file=img_info["cosar_filename"],
            sicd_file=img_info["sicd_filename"],
            classification=config.classification,
            chan_index=img_info["chan_index"],
            tx_polarizations=tx_polarizations,
            tx_rcv_pols=tx_rcv_pols,
        )


if __name__ == "__main__":
    main()
