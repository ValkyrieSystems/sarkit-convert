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
import datetime
import pathlib

import astropy.coordinates as apcoord
import astropy.units as apu
import astropy.utils
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
import scipy.spatial.transform

from sarkit_convert import _utils as utils

astropy.utils.iers.conf.autodownload = False

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
    apc_poly = utils.fit_state_vectors(
        (0, (stop_time - start_time).total_seconds()),
        times,
        position,
        velocity,
        acceleration,
        order=5,
    )

    return apc_poly


def hdf5_to_sicd(
    h5_filename,
    sicd_filename,
    classification,
    ostaid,
    img_str,
    chan_index,
    tx_polarizations,
    tx_rcv_pols,
):
    with h5py.File(h5_filename, "r") as h5file:
        h5_attrs = extract_attributes(h5file)
        mission_id = h5_attrs["Mission ID"]
        if mission_id == "CSG":
            dataset_str = "IMG"
            burst_str = "B0001"
        else:
            dataset_str = "SBI"
            burst_str = "B001"
        sample_data_h5_path = f"{img_str}/{dataset_str}"
        sample_data_shape = h5file[sample_data_h5_path].shape
        sample_data_dtype = h5file[sample_data_h5_path].dtype

    # Timeline
    ref_time = dateutil.parser.parse(h5_attrs["Reference UTC"])
    collection_start_time = dateutil.parser.parse(h5_attrs["Scene Sensing Start UTC"])
    collection_stop_time = dateutil.parser.parse(h5_attrs["Scene Sensing Stop UTC"])
    collection_duration = (collection_stop_time - collection_start_time).total_seconds()
    prf = h5_attrs[img_str]["PRF"]
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
    tx_pulse_length = h5_attrs[img_str]["Range Chirp Length"]
    tx_fm_rate = h5_attrs[img_str]["Range Chirp Rate"]
    tx_rf_bw = np.abs(tx_fm_rate * tx_pulse_length)
    tx_freq_min = center_frequency - 0.5 * tx_rf_bw
    tx_freq_max = center_frequency + 0.5 * tx_rf_bw
    tx_freq_start = center_frequency - (tx_pulse_length / 2 * tx_fm_rate)
    adc_sample_rate = h5_attrs[img_str]["Sampling Rate"]
    rcv_window_length = (
        h5_attrs[img_str]["Echo Sampling Window Length"] / adc_sample_rate
    )
    tx_rcv_polarization = tx_rcv_pols[chan_index - 1]
    tx_polarization = tx_rcv_polarization[0]

    # Grid
    assert h5_attrs["Lines Order"] == "EARLY-LATE"
    assert h5_attrs["Columns Order"] == "NEAR-FAR"
    spacings = np.array(
        [
            h5_attrs[img_str][dataset_str]["Column Spacing"],
            h5_attrs[img_str][dataset_str]["Line Spacing"],
        ]
    )
    intervals = np.array(
        [
            h5_attrs[img_str][dataset_str]["Column Time Interval"],
            h5_attrs[img_str][dataset_str]["Line Time Interval"],
        ]
    )
    zd_az_0 = h5_attrs[img_str][dataset_str]["Zero Doppler Azimuth First Time"]
    zd_rg_0 = h5_attrs[img_str][dataset_str]["Zero Doppler Range First Time"]
    row_bw = (
        h5_attrs[img_str]["Range Focusing Bandwidth"]
        * 2
        / scipy.constants.speed_of_light
    )
    row_wid = 1 / row_bw
    col_bw = (
        min(
            h5_attrs[img_str]["Azimuth Focusing Transition Bandwidth"] * intervals[1], 1
        )
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
        range_ref = h5_attrs[img_str]["Range Polynomial Reference Time"]
        azimuth_ref = h5_attrs[img_str]["Azimuth Polynomial Reference Time"]
        azimuth_ref_zd = h5_attrs[img_str]["Azimuth Polynomial Reference Time - ZD"]
        azimuth_first_time = h5_attrs[img_str]["B0001"]["Azimuth First Time"]
        azimuth_last_time = h5_attrs[img_str]["B0001"]["Azimuth Last Time"]
        raw_times = np.linspace(azimuth_first_time, azimuth_last_time, num_grid_pts)

        centroid_range_poly = h5_attrs[img_str][
            "Doppler Centroid vs Range Time Polynomial"
        ]
        centroid_azimuth_poly = h5_attrs[img_str][
            "Doppler Centroid vs Azimuth Time Polynomial - RAW"
        ]
        raw_doppler_centroid = npp.polyval(
            raw_times - azimuth_ref, centroid_azimuth_poly
        )

        rate_range_poly = h5_attrs[img_str]["Doppler Rate vs Range Time Polynomial"]
        rate_azimuth_poly = h5_attrs[img_str]["Doppler Rate vs Azimuth Time Polynomial"]
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
    doppler_centroid_poly = utils.polyfit2d(
        grid_coords[..., 0].flatten(),
        grid_coords[..., 1].flatten(),
        doppler_centroid.flatten(),
        4,
        4,
    )
    doppler_rate_poly = utils.polyfit2d(
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
    time_coa_poly = utils.polyfit2d(
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
    drsf_poly = utils.polyfit2d(
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
    scp_set = sksicd.projection.ProjectionSetsMono(
        t_COA=np.array([scp_tcoa]),
        ARP_COA=np.array([npp.polyval(scp_tcoa, apc_poly)]),
        VARP_COA=np.array([npp.polyval(scp_tcoa, npp.polyder(apc_poly))]),
        R_COA=np.array([scp_rcoa]),
        Rdot_COA=np.array([scp_rratecoa]),
    )
    scp_ecf, _, _ = sksicd.projection.r_rdot_to_constant_hae_surface(
        look, sc_ecf, scp_set, scp_hae
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

    # Antenna
    attitude_quaternion = np.roll(h5_attrs["Attitude Quaternions"], -1, axis=1)
    attitude_times = h5_attrs["Attitude Times"]
    attitude_utcs = [
        ref_time + datetime.timedelta(seconds=attitude_time)
        for attitude_time in attitude_times
    ]

    inertial_position = h5_attrs["Inertial Satellite Position"]
    inertial_velocity = h5_attrs["Inertial Satellite Velocity"]
    inertial_acceleration = h5_attrs["Inertial Satellite Acceleration"]
    eci_apc_poly = utils.fit_state_vectors(
        (0, (collection_stop_time - collection_start_time).total_seconds()),
        h5_attrs["State Vectors Times"]
        - (collection_start_time - ref_time).total_seconds(),
        inertial_position,
        inertial_velocity,
        inertial_acceleration,
        order=5,
    )

    def get_nadir_plane_at_time(time):
        obstime = collection_start_time + datetime.timedelta(seconds=time)
        eci_pos = npp.polyval(time, eci_apc_poly)
        eci_vel = npp.polyval(time, npp.polyder(eci_apc_poly))

        # Derived from https://adsabs.harvard.edu/full/2006ESASP.606E..35C, Section 3
        z_sc = -eci_pos / np.linalg.norm(eci_pos)
        y_sc_dir = np.cross(z_sc, eci_vel)
        y_sc = y_sc_dir / np.linalg.norm(y_sc_dir)
        x_sc = np.cross(y_sc, z_sc)
        eci_2_ecf = np.array(
            [
                apcoord.GCRS(
                    apcoord.CartesianRepresentation(1, 0, 0, unit=apu.m),
                    obstime=obstime,
                )
                .transform_to(apcoord.ITRS(obstime=obstime))
                .data.xyz.value,
                apcoord.GCRS(
                    apcoord.CartesianRepresentation(0, 1, 0, unit=apu.m),
                    obstime=obstime,
                )
                .transform_to(apcoord.ITRS(obstime=obstime))
                .data.xyz.value,
                apcoord.GCRS(
                    apcoord.CartesianRepresentation(0, 0, 1, unit=apu.m),
                    obstime=obstime,
                )
                .transform_to(apcoord.ITRS(obstime=obstime))
                .data.xyz.value,
            ]
        )
        x_sc = x_sc @ eci_2_ecf
        y_sc = y_sc @ eci_2_ecf
        z_sc = z_sc @ eci_2_ecf
        return np.array([x_sc, y_sc, z_sc])

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
    good_att_quat = attitude_quaternion[good_indices]
    nadir_planes = [get_nadir_plane_at_time(time) for time in good_times]
    body_frame = np.array(
        [
            scipy.spatial.transform.Rotation.from_quat(att_quat)
            .inv()
            .apply(nadir_plane.T)
            .T
            for att_quat, nadir_plane in zip(good_att_quat, nadir_planes)
        ]
    )

    ux_rot = body_frame[:, 0, :]
    uy_rot = body_frame[:, 1, :]

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

    freq_zero = h5_attrs["Radar Frequency"]
    antenna_beam_elevation = h5_attrs[img_str]["Antenna Beam Elevation"]
    fit_order = 4

    def fit_steering(code_change_lines, dcs):
        if len(code_change_lines) > 1:
            times = code_change_lines / prf
            return npp.polyfit(times, dcs, fit_order)
        else:
            return np.array(dcs).reshape((1,))

    if radar_mode_type != "STRIPMAP":
        azimuth_ramp_code_change_lines = h5_attrs[img_str][burst_str][
            "Azimuth Ramp Code Change Lines"
        ]
        azimuth_steering = h5_attrs[img_str][burst_str]["Azimuth Steering"]
        elevation_ramp_code_change_lines = h5_attrs[img_str][burst_str][
            "Elevation Ramp Code Change Lines"
        ]
        elevation_steering = h5_attrs[img_str][burst_str]["Elevation Steering"]
        eb_dcx = np.sin(np.deg2rad(azimuth_steering))
        eb_dcx_poly = fit_steering(azimuth_ramp_code_change_lines, eb_dcx)
        eb_dcy = -np.sin(np.deg2rad(antenna_beam_elevation + elevation_steering))
        eb_dcy_poly = fit_steering(elevation_ramp_code_change_lines, eb_dcy)
    else:
        eb_dcx_poly = [0.0]
        eb_dcy_poly = [-np.sin(np.deg2rad(antenna_beam_elevation))]

    antenna_az_gains = h5_attrs[img_str]["Azimuth Antenna Pattern Gains"]
    antenna_az_origin = h5_attrs[img_str]["Azimuth Antenna Pattern Origin"]
    antenna_az_spacing = h5_attrs[img_str]["Azimuth Antenna Pattern Resolution"]

    antenna_rg_gains = h5_attrs[img_str]["Range Antenna Pattern Gains"]
    antenna_rg_origin = h5_attrs[img_str]["Range Antenna Pattern Origin"]
    antenna_rg_spacing = h5_attrs[img_str]["Range Antenna Pattern Resolution"]

    def fit_gains(origin, spacing, gains):
        fit_limit = -9
        array_mask = gains > fit_limit
        dcs = np.sin(np.deg2rad(origin + spacing * np.arange(len(gains))))
        return npp.polyfit(dcs[array_mask], gains[array_mask], fit_order)

    antenna_array_gain = np.zeros((fit_order + 1, fit_order + 1), dtype=float)
    antenna_array_gain[0, :] = fit_gains(
        antenna_rg_origin, antenna_rg_spacing, antenna_rg_gains
    )
    antenna_array_gain[:, 0] = fit_gains(
        antenna_az_origin, antenna_az_spacing, antenna_az_gains
    )
    antenna_array_gain[0, 0] = 0.0

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
    rcs_row_sf = None
    rcs_col_sf = None
    if row_window_name == "HAMMING":
        wgts = scipy.signal.windows.general_hamming(512, row_window_coeff, sym=True)
        wgtfunc = sicd.WgtFunct()
        sksicd.TRANSCODERS["Grid/Row/WgtFunct"].set_elem(wgtfunc, wgts)
        grid.find("./{*}Row").append(wgtfunc)
        row_broadening_factor = utils.broadening_from_amp(wgts)
        row_wid = row_broadening_factor / row_bw
        sksicd.DblType().set_elem(grid.find("./{*}Row/{*}ImpRespWid"), row_wid)
        rcs_row_sf = 1 + np.var(wgts) / np.mean(wgts) ** 2
    if col_window_name == "HAMMING":
        wgts = scipy.signal.windows.general_hamming(512, col_window_coeff, sym=True)
        wgtfunc = sicd.WgtFunct()
        sksicd.TRANSCODERS["Grid/Col/WgtFunct"].set_elem(wgtfunc, wgts)
        grid.find("./{*}Col").append(wgtfunc)
        col_broadening_factor = utils.broadening_from_amp(wgts)
        col_wid = col_broadening_factor / col_bw
        sksicd.DblType().set_elem(grid.find("./{*}Col/{*}ImpRespWid"), col_wid)
        rcs_col_sf = 1 + np.var(wgts) / np.mean(wgts) ** 2

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

    rcv_channels = sicd.RcvChannels(
        {"size": str(len(tx_rcv_pols))},
    )
    for ndx, tx_rcv_pol in enumerate(tx_rcv_pols):
        rcv_channels.append(
            sicd.ChanParameters(
                {"index": str(ndx + 1)}, sicd.TxRcvPolarization(tx_rcv_pol)
            )
        )

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
        rcv_channels,
    )
    if len(tx_polarizations) > 1:
        radar_collection.find("./{*}TxPolarization").text = "SEQUENCE"
        tx_sequence = sicd.TxSequence({"size": str(len(tx_polarizations))})
        for ndx, tx_pol in enumerate(tx_polarizations):
            tx_sequence.append(
                sicd.TxStep({"index": str(ndx + 1)}, sicd.TxPolarization(tx_pol))
            )
        rcv_channels.addprevious(tx_sequence)

    image_formation = sicd.ImageFormation(
        sicd.RcvChanProc(sicd.NumChanProc("1"), sicd.ChanIndex(str(chan_index))),
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

    antenna = sicd.Antenna(
        sicd.TwoWay(
            sicd.XAxisPoly(),
            sicd.YAxisPoly(),
            sicd.FreqZero(str(freq_zero)),
            sicd.EB(
                sicd.DCXPoly(),
                sicd.DCYPoly(),
            ),
            sicd.Array(
                sicd.GainPoly(),
                sicd.PhasePoly(),
            ),
        ),
    )
    sksicd.XyzPolyType().set_elem(
        antenna.find("./{*}TwoWay/{*}XAxisPoly"), ant_x_dir_poly
    )
    sksicd.XyzPolyType().set_elem(
        antenna.find("./{*}TwoWay/{*}YAxisPoly"), ant_y_dir_poly
    )
    sksicd.PolyType().set_elem(
        antenna.find("./{*}TwoWay/{*}EB/{*}DCXPoly"), eb_dcx_poly
    )
    sksicd.PolyType().set_elem(
        antenna.find("./{*}TwoWay/{*}EB/{*}DCYPoly"), eb_dcy_poly
    )
    sksicd.Poly2dType().set_elem(
        antenna.find("./{*}TwoWay/{*}Array/{*}GainPoly"), antenna_array_gain
    )
    sksicd.Poly2dType().set_elem(
        antenna.find("./{*}TwoWay/{*}Array/{*}PhasePoly"),
        np.zeros(dtype=float, shape=(1, 1)),
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
        antenna,
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
                cal = h5_attrs[img_str]["Calibration Constant"]
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
            if rcs_row_sf and rcs_col_sf:
                rcssf_poly = betazero_poly * (
                    rcs_row_sf * rcs_col_sf / (row_bw * col_bw)
                )
                radiometric.find("./{*}SigmaZeroSFPoly").addprevious(sicd.RCSSFPoly())
                sksicd.Poly2dType().set_elem(
                    radiometric.find("./{*}RCSSFPoly"), rcssf_poly
                )
            sicd_xml_obj.find("./{*}Antenna").addprevious(radiometric)

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
        "output_sicd_file",
        type=pathlib.Path,
        help='path of the output SICD file. The string "{pol}" will be replaced with polarization for multiple images',
    )
    parser.add_argument(
        "--ostaid",
        help="content of the originating station ID (OSTAID) field of the NITF header",
        default="Unknown",
    )
    config = parser.parse_args(args)

    tx_polarizations = []
    with h5py.File(config.input_h5_file, "r") as h5file:
        acquisition_mode = h5file.attrs["Acquisition Mode"].decode()
        if "scan" in acquisition_mode.lower():
            raise ValueError("ScanSar modes not supported")
        mission_id = h5file.attrs["Mission ID"].decode()
        images = dict()
        if mission_id == "CSG":
            img_str = "S01"
            polarization = h5file.attrs["Polarization"].decode()
            filename = pathlib.Path(
                str(config.output_sicd_file).format(pol=polarization)
            )
            images[img_str] = {
                "polarization": polarization,
                "chan_index": 1,
                "filename": filename,
            }
            tx_polarizations.append(polarization[0])
            tx_rcv_pols = [f"{polarization[0]}:{polarization[1]}"]
        else:
            img_ndx = 1
            images = dict()
            tx_rcv_pols = []
            while (img_str := f"S{img_ndx:02}") in h5file:
                polarization = h5file[img_str].attrs["Polarisation"].decode()
                filename = pathlib.Path(
                    str(config.output_sicd_file).format(pol=polarization)
                )
                images[img_str] = {
                    "polarization": polarization,
                    "chan_index": img_ndx,
                    "filename": filename,
                }
                tx_rcv_pols.append(f"{polarization[0]}:{polarization[1]}")
                if (tx_polarization := polarization[0]) not in tx_polarizations:
                    tx_polarizations.append(tx_polarization)
                img_ndx += 1

    if len(images) != len(set([image["filename"] for image in images.values()])):
        raise ValueError("Output filename does not include necessary polarization slug")

    for img_str, img_info in images.items():
        hdf5_to_sicd(
            h5_filename=config.input_h5_file,
            sicd_filename=img_info["filename"],
            classification=config.classification,
            ostaid=config.ostaid,
            img_str=img_str,
            chan_index=img_info["chan_index"],
            tx_polarizations=tx_polarizations,
            tx_rcv_pols=tx_rcv_pols,
        )


if __name__ == "__main__":
    main()
