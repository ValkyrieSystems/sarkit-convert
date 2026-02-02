"""
=====================
Sentinel SAFE to SICD
=====================

Convert a complex image(s) from the Sentinel SAFE to SICD(s).

"""

import argparse
import contextlib
import datetime
import functools
import pathlib

import dateutil.parser
import lxml.etree
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import sarkit.verification
import sarkit.wgs84
import scipy.interpolate
import scipy.signal
from sarkit import _constants
from tifffile import tifffile

from sarkit_convert import __version__
from sarkit_convert import _utils as utils

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}


def _get_file_sets(safe_product_folder, root_node):
    """Extracts paths for measurement and metadata files from a Sentinel manifest.safe file."""

    def _get_file_location(root_node, schema_type, fids):
        if isinstance(fids, str):
            fids = [
                fids,
            ]
        for tid in fids:
            data_object = root_node.find(
                f"dataObjectSection/dataObject[@repID='{schema_type}']/[@ID='{tid}']",
            )
            if data_object is None:
                continue
            return (
                safe_product_folder
                / data_object.find("./byteStream/fileLocation").attrib["href"]
            )
        return None

    files = []
    for mdu in root_node.findall(
        "./informationPackageMap/{*}contentUnit/{*}contentUnit/[@repID='s1Level1MeasurementSchema']",
    ):
        # get the data file for this measurement
        fnames = {
            "data": _get_file_location(
                root_node,
                "s1Level1MeasurementSchema",
                mdu.find("dataObjectPointer").attrib["dataObjectID"],
            ),
        }
        # get the ids for product, noise, and calibration associated with this measurement data unit
        ids = mdu.attrib["dmdID"].split()
        # translate these ids to data object ids=file ids for the data files
        fids = [
            root_node.find(
                f"./metadataSection/metadataObject[@ID='{did}']/dataObjectPointer"
            ).attrib["dataObjectID"]
            for did in ids
        ]
        # NB: there is (at most) one of these per measurement data unit
        fnames["product"] = _get_file_location(root_node, "s1Level1ProductSchema", fids)
        fnames["noise"] = _get_file_location(root_node, "s1Level1NoiseSchema", fids)
        fnames["calibration"] = _get_file_location(
            root_node, "s1Level1CalibrationSchema", fids
        )
        files.append(fnames)
    return files


def _get_slice(product_root_node):
    slice_number = product_root_node.find(
        "./imageAnnotation/imageInformation/sliceNumber"
    )
    if slice_number is None:
        return "0"
    else:
        return slice_number.text


def _get_swath(product_root_node):
    return product_root_node.find("./adsHeader/swath").text


def _compute_arp_poly_coefs(root_node, start):
    orbit_list = root_node.findall("./generalAnnotation/orbitList/orbit")
    shp = (len(orbit_list),)
    t_s = np.empty(shp, dtype=np.float64)
    x_s = np.empty(shp, dtype=np.float64)
    y_s = np.empty(shp, dtype=np.float64)
    z_s = np.empty(shp, dtype=np.float64)
    for j, orbit in enumerate(orbit_list):
        t_s[j] = (
            dateutil.parser.parse(orbit.find("./time").text) - start
        ).total_seconds()
        x_s[j] = float(orbit.find("./position/x").text)
        y_s[j] = float(orbit.find("./position/y").text)
        z_s[j] = float(orbit.find("./position/z").text)

    poly_order = min(5, t_s.size - 1)
    p_x = npp.polyfit(t_s, x_s, poly_order)
    p_y = npp.polyfit(t_s, y_s, poly_order)
    p_z = npp.polyfit(t_s, z_s, poly_order)
    return np.stack((p_x, p_y, p_z))


def _collect_base_info(root_node):
    # Collection Info
    base_info = dict()
    platform = root_node.find(
        "./metadataSection/metadataObject[@ID='platform']/metadataWrap/xmlData/{*}platform",
    )
    base_info["collector_name"] = (
        platform.find("{*}familyName").text + platform.find("{*}number").text
    )
    base_info["collect_type"] = "MONOSTATIC"
    mode_id = platform.find(
        "./{*}instrument/{*}extension/{*}instrumentMode/{*}mode"
    ).text
    if mode_id == "SM":
        base_info["mode_type"] = "STRIPMAP"
    else:
        # TOPSAR - closest SICD analog is Dynamic Stripmap
        base_info["mode_type"] = "DYNAMIC STRIPMAP"

    relative_orbit_num = int(
        root_node.findtext(".//{*}relativeOrbitNumber[@type='start']")
    )
    base_info["parameters"] = {
        "RELATIVE_ORBIT_NUM": relative_orbit_num,
    }

    base_info["start_time_anx"] = float(
        root_node.findtext(".//{*}acquisitionPeriod//{*}startTimeANX")
    )
    base_info["acq_start_str"] = root_node.findtext(
        ".//{*}acquisitionPeriod//{*}startTime"
    )

    # Image Creation
    processing = root_node.find(
        "./metadataSection/metadataObject[@ID='processing']/metadataWrap/xmlData/{*}processing",
    )
    facility = processing.find("{*}facility")
    software = facility.find("{*}software")
    base_info["creation_application"] = (
        f"{software.attrib['name']} {software.attrib['version']}"
    )
    base_info["creation_date_time"] = dateutil.parser.parse(processing.attrib["stop"])

    # Radar Collection
    polarizations = root_node.findall(
        "./metadataSection/metadataObject[@ID='generalProductInformation']/metadataWrap/xmlData/{*}standAloneProductInformation/{*}transmitterReceiverPolarisation",
    )

    base_info["tx_rcv_polarization"] = []
    for pol in polarizations:
        base_info["tx_rcv_polarization"].append(f"{pol.text[0]}:{pol.text[1]}")

    return base_info


@functools.lru_cache
def _window_params(window_name, coefficient):
    window_name = window_name.upper()
    if window_name == "HAMMING":
        wgts = scipy.signal.windows.general_hamming(512, float(coefficient), sym=True)
    elif window_name == "KAISER":
        wgts = scipy.signal.windows.kaiser(512, float(coefficient), sym=True)
    else:  # Default to UNIFORM
        window_name = "UNIFORM"
        coefficient = None
        wgts = np.ones(256)

    broadening_factor = utils.broadening_from_amp(wgts)

    return {
        "window_name": window_name,
        "coefficient": coefficient,
        "wgts": wgts,
        "broadening_factor": broadening_factor,
    }


def _collect_swath_info(product_root_node):
    swath_info = dict()
    burst_list = product_root_node.findall("./swathTiming/burstList/burst")

    # Collection Info
    swath_info["collector_name"] = product_root_node.find("./adsHeader/missionId").text
    swath_info["mode_id"] = product_root_node.find("./adsHeader/mode").text
    t_slice = _get_slice(product_root_node)
    swath = _get_swath(product_root_node)

    mission_data_take_id = product_root_node.find("./adsHeader/missionDataTakeId").text
    swath_info["parameters"] = {
        "SLICE": t_slice,
        "SWATH": swath,
        "ORBIT_SOURCE": "SLC_INTERNAL",
        "MISSION_DATA_TAKE_ID": mission_data_take_id,
    }

    slice_list = product_root_node.findall(
        "./imageAnnotation/imageInformation/sliceList/slice"
    )
    swath_info["sensing_start"] = dateutil.parser.parse(
        slice_list[int(t_slice) - 1].find("./sensingStartTime").text
    )
    swath_info["sensing_stop"] = dateutil.parser.parse(
        slice_list[int(t_slice) - 1].find("./sensingStopTime").text
    )

    # Radar Collection
    center_frequency = float(
        product_root_node.find(
            "./generalAnnotation/productInformation/radarFrequency"
        ).text
    )
    swath_info["tx_freq_start"] = center_frequency + float(
        product_root_node.find(
            "./generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseStartFrequency"
        ).text
    )
    swath_info["tx_pulse_length"] = float(
        product_root_node.find(
            "./generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseLength"
        ).text
    )
    swath_info["tx_fm_rate"] = float(
        product_root_node.find(
            "./generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseRampRate"
        ).text
    )
    swath_info["tx_rf_bw"] = swath_info["tx_pulse_length"] * swath_info["tx_fm_rate"]
    pol = product_root_node.find("./adsHeader/polarisation").text
    swath_info["tx_polarization"] = pol[0]
    swath_info["rcv_polarization"] = pol[1]
    swath_info["tx_freq"] = (
        swath_info["tx_freq_start"],
        swath_info["tx_freq_start"] + swath_info["tx_rf_bw"],
    )
    swath_info["adc_sample_rate"] = float(
        product_root_node.find(
            "./generalAnnotation/productInformation/rangeSamplingRate"
        ).text
    )
    swl_list = product_root_node.findall(
        "./generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/swlList/swl"
    )
    swath_info["rcv_window_length"] = [
        float(swl.find("./value").text) for swl in swl_list
    ]

    # Timeline
    swath_info["prf"] = float(
        product_root_node.find(
            "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"
        ).text
    )
    swath_info["ipp_poly"] = (0, swath_info["prf"])

    # Image Formation
    swath_info["tx_rcv_polarization_proc"] = (
        f"{swath_info['tx_polarization']}:{swath_info['rcv_polarization']}"
    )
    swath_info["image_form_algo"] = "RMA"
    swath_info["tx_freq_proc"] = swath_info["tx_freq"]
    swath_info["image_beam_comp"] = "SV"
    swath_info["az_autofocus"] = "NO"
    swath_info["rg_autofocus"] = "NO"
    swath_info["st_beam_comp"] = "GLOBAL" if swath_info["mode_id"][0] == "S" else "SV"

    # Image Data
    pixel_value = product_root_node.find(
        "./imageAnnotation/imageInformation/pixelValue"
    ).text
    output_pixels = product_root_node.find(
        "./imageAnnotation/imageInformation/outputPixels"
    ).text
    if pixel_value == "Complex" and output_pixels == "16 bit Signed Integer":
        swath_info["pixel_type"] = "RE16I_IM16I"
    else:
        raise ValueError(
            f"SLC data should be 16-bit complex, got pixelValue = {pixel_value} and outputPixels = {output_pixels}."
        )
    if len(burst_list) > 0:
        # TOPSAR
        swath_info["num_rows"] = int(
            product_root_node.find("./swathTiming/samplesPerBurst").text
        )
        swath_info["num_cols"] = int(
            product_root_node.find("./swathTiming/linesPerBurst").text
        )
    else:
        # STRIPMAP
        swath_info["num_rows"] = int(
            product_root_node.find(
                "./imageAnnotation/imageInformation/numberOfSamples"
            ).text
        )
        swath_info["num_cols"] = int(
            product_root_node.find(
                "./imageAnnotation/imageInformation/numberOfLines"
            ).text
        )
    swath_info["first_row"] = 0
    swath_info["first_col"] = 0
    swath_info["scp_pixel"] = (
        (swath_info["num_rows"] - 1) // 2,
        (swath_info["num_cols"] - 1) // 2,
    )

    # RMA
    swath_info["freq_zero"] = center_frequency
    swath_info["dop_centroid_coa"] = "true"
    tau_0 = float(
        product_root_node.find("./imageAnnotation/imageInformation/slantRangeTime").text
    )
    delta_tau_s = 1.0 / float(
        product_root_node.find(
            "./generalAnnotation/productInformation/rangeSamplingRate"
        ).text
    )
    swath_info["r_ca_scp"] = (0.5 * _constants.speed_of_light) * (
        tau_0 + swath_info["scp_pixel"][0] * delta_tau_s
    )
    swath_info["rm_algo_type"] = "RG_DOP"
    swath_info["image_type"] = "INCA"

    # Grid
    swath_info["image_plane"] = (
        "SLANT"
        if product_root_node.find(
            "./generalAnnotation/productInformation/projection"
        ).text
        == "Slant Range"
        else None
    )
    swath_info["grid_type"] = "RGZERO"
    range_proc = product_root_node.find(
        "./imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing"
    )

    window_info = _window_params(
        range_proc.find("./windowType").text,
        range_proc.find("./windowCoefficient").text,
    )
    swath_info["row_window_name"] = window_info["window_name"]
    swath_info["row_params"] = window_info["coefficient"]
    swath_info["row_wgts"] = window_info["wgts"]
    swath_info["row_broadening_factor"] = window_info["broadening_factor"]

    swath_info["row_ss"] = (_constants.speed_of_light / 2) * delta_tau_s
    swath_info["row_sgn"] = -1
    swath_info["row_kctr"] = 2 * center_frequency / _constants.speed_of_light
    swath_info["row_imp_res_bw"] = (
        2.0
        * float(range_proc.find("./processingBandwidth").text)
        / _constants.speed_of_light
    )
    swath_info["row_deltak_coa_poly"] = np.array([[0]])

    az_proc = product_root_node.find(
        "./imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing"
    )
    swath_info["col_ss"] = float(
        product_root_node.find(
            "./imageAnnotation/imageInformation/azimuthPixelSpacing"
        ).text
    )
    dop_bw = float(az_proc.find("./processingBandwidth").text)
    swath_info["ss_zd_s"] = float(
        product_root_node.find(
            "./imageAnnotation/imageInformation/azimuthTimeInterval"
        ).text
    )

    window_info = _window_params(
        az_proc.find("./windowType").text, az_proc.find("./windowCoefficient").text
    )
    swath_info["col_window_name"] = window_info["window_name"]
    swath_info["col_params"] = window_info["coefficient"]
    swath_info["col_wgts"] = window_info["wgts"]
    swath_info["col_broadening_factor"] = window_info["broadening_factor"]

    swath_info["col_sgn"] = -1
    swath_info["col_kctr"] = 0.0
    swath_info["col_imp_res_bw"] = dop_bw * swath_info["ss_zd_s"] / swath_info["col_ss"]

    swath_info["row_imp_res_wid"] = (
        swath_info["row_broadening_factor"] / swath_info["row_imp_res_bw"]
    )
    swath_info["col_imp_res_wid"] = (
        swath_info["col_broadening_factor"] / swath_info["col_imp_res_bw"]
    )

    return swath_info


def _collect_burst_info(product_root_node, base_info, swath_info):
    burst_list = product_root_node.findall("./swathTiming/burstList/burst")
    # parse the geolocation information - for SCP calculation
    geo_grid_point_list = product_root_node.findall(
        "./geolocationGrid/geolocationGridPointList/geolocationGridPoint"
    )
    geo_pixels = np.zeros((len(geo_grid_point_list), 2), dtype=np.float64)
    geo_coords_llh = np.zeros((len(geo_grid_point_list), 3), dtype=np.float64)
    for i, grid_point in enumerate(geo_grid_point_list):
        geo_pixels[i, :] = (
            float(grid_point.find("./pixel").text),
            float(grid_point.find("./line").text),
        )
        geo_coords_llh[i, :] = (
            float(grid_point.find("./latitude").text),
            float(grid_point.find("./longitude").text),
            float(grid_point.find("./height").text),
        )
    geo_coords_ecf = sarkit.wgs84.geodetic_to_cartesian(geo_coords_llh)

    def _shift(coefs, t_0: float, alpha: float = 1):
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

    def _calc_deltaks(x_coords, y_coords, deltak_coa_poly, imp_resp_bw, spacing):
        """Calculate the minimum and maximum DeltaK values"""
        deltaks = npp.polyval2d(x_coords, y_coords, deltak_coa_poly)
        min_deltak = np.amin(deltaks) - 0.5 * imp_resp_bw
        max_deltak = np.amax(deltaks) + 0.5 * imp_resp_bw

        if (min_deltak < -0.5 / abs(spacing)) or (max_deltak > 0.5 / abs(spacing)):
            min_deltak = -0.5 / abs(spacing)
            max_deltak = -min_deltak

        return min_deltak, max_deltak

    def _get_scps(swath_info, count):
        # SCPPixel - points at which to interpolate geo_pixels & geo_coords data
        num_rows = swath_info["num_rows"]
        num_cols = swath_info["num_cols"]
        scp_pixels = np.zeros((count, 2), dtype=np.float64)
        scp_pixels[:, 0] = int((num_rows - 1) / 2.0)
        scp_pixels[:, 1] = int((num_cols - 1) / 2.0) + num_cols * (
            np.arange(count, dtype=np.float64)
        )
        scps = np.zeros((count, 3), dtype=np.float64)

        for j in range(3):
            scps[:, j] = scipy.interpolate.griddata(
                geo_pixels, geo_coords_ecf[:, j], scp_pixels
            )
        return scps

    def _calc_rma_and_grid_info(
        swath_info, burst_info, first_line_relative_start, start
    ):
        # set TimeCAPoly
        scp_pixel = swath_info["scp_pixel"]
        eta_mid = swath_info["ss_zd_s"] * scp_pixel[1]
        row_ss = swath_info["row_ss"]
        col_ss = swath_info["col_ss"]

        time_ca_poly_coefs = [
            first_line_relative_start + eta_mid,
            swath_info["ss_zd_s"] / col_ss,
        ]
        burst_info["time_ca_poly_coefs"] = time_ca_poly_coefs
        r_ca_scp = swath_info["r_ca_scp"]
        range_time_scp = r_ca_scp * 2 / _constants.speed_of_light
        # get velocity polynomial
        arp_poly = burst_info["arp_poly_coefs"]
        vel_poly = npp.polyder(arp_poly)
        # We pick a single velocity magnitude at closest approach to represent
        # the entire burst.  This is valid, since the magnitude of the velocity
        # changes very little.

        vm_ca = np.linalg.norm(npp.polyval(time_ca_poly_coefs[0], vel_poly))
        azimuth_fm_rate_list = product_root_node.findall(
            "./generalAnnotation/azimuthFmRateList/azimuthFmRate"
        )
        shp = (len(azimuth_fm_rate_list),)
        az_rate_times = np.empty(shp, dtype=np.float64)
        az_rate_t0 = np.empty(shp, dtype=np.float64)
        k_a_poly = []
        for j, az_fm_rate in enumerate(azimuth_fm_rate_list):
            az_rate_times[j] = (
                dateutil.parser.parse(az_fm_rate.find("./azimuthTime").text) - start
            ).total_seconds()
            az_rate_t0[j] = float(az_fm_rate.find("./t0").text)
            if az_fm_rate.find("c0") is not None:
                k_a_poly.append(
                    np.array(
                        [
                            float(az_fm_rate.find("./c0").text),
                            float(az_fm_rate.find("./c1").text),
                            float(az_fm_rate.find("./c2").text),
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                k_a_poly.append(
                    np.fromstring(
                        az_fm_rate.find("./azimuthFmRatePolynomial").text, sep=" "
                    )
                )

        # find the closest fm rate polynomial
        az_rate_poly_ind = int(np.argmin(np.abs(az_rate_times - time_ca_poly_coefs[0])))
        az_rate_poly_coefs = k_a_poly[az_rate_poly_ind]
        dr_ca_poly = _shift(
            az_rate_poly_coefs,
            t_0=az_rate_t0[az_rate_poly_ind] - range_time_scp,
            alpha=2 / _constants.speed_of_light,
        )
        r_ca = np.array([r_ca_scp, 1], dtype=np.float64)
        burst_info["drsf_poly_coefs"] = np.reshape(
            -np.convolve(dr_ca_poly, r_ca)
            * (
                _constants.speed_of_light
                / (2 * swath_info["freq_zero"] * vm_ca * vm_ca)
            ),
            (-1, 1),
        )

        # Doppler Centroid
        dc_estimate_list = product_root_node.findall(
            "./dopplerCentroid/dcEstimateList/dcEstimate"
        )
        shp = (len(dc_estimate_list),)
        dc_est_times = np.empty(shp, dtype=np.float64)
        dc_t0 = np.empty(shp, dtype=np.float64)
        data_dc_poly = []
        for j, dc_estimate in enumerate(dc_estimate_list):
            dc_est_times[j] = (
                dateutil.parser.parse(dc_estimate.find("./azimuthTime").text) - start
            ).total_seconds()
            dc_t0[j] = float(dc_estimate.find("./t0").text)
            data_dc_poly.append(
                np.fromstring(dc_estimate.find("./dataDcPolynomial").text, sep=" ")
            )
        # find the closest doppler centroid polynomial
        dc_poly_ind = int(np.argmin(np.abs(dc_est_times - time_ca_poly_coefs[0])))
        # we are going to move the respective polynomial from reference point as dc_t0 to
        # reference point at SCP time.
        dc_poly_coefs = data_dc_poly[dc_poly_ind]
        # Fit DeltaKCOAPoly, DopCentroidPoly, and TimeCOAPoly from data
        tau_0 = float(
            product_root_node.find(
                "./imageAnnotation/imageInformation/slantRangeTime"
            ).text
        )
        delta_tau_s = 1.0 / float(
            product_root_node.find(
                "./generalAnnotation/productInformation/rangeSamplingRate"
            ).text
        )

        # common use for the fitting efforts
        poly_order = 4
        grid_samples = poly_order + 4
        num_rows = swath_info["num_rows"]
        num_cols = swath_info["num_cols"]
        first_row = swath_info["first_row"]
        first_col = swath_info["first_col"]
        cols = np.linspace(0, num_cols - 1, grid_samples, dtype=np.int64)
        rows = np.linspace(0, num_rows - 1, grid_samples, dtype=np.int64)
        coords_az = (cols - scp_pixel[1] + first_col) * col_ss
        coords_rg = (rows - scp_pixel[0] + first_row) * row_ss
        coords_az_2d, coords_rg_2d = np.meshgrid(coords_az, coords_rg)

        # fit DeltaKCOAPoly
        tau = tau_0 + delta_tau_s * rows
        # Azimuth steering rate (constant, not dependent on burst or range)
        k_psi = np.deg2rad(
            float(
                product_root_node.find(
                    "./generalAnnotation/productInformation/azimuthSteeringRate"
                ).text
            )
        )
        k_s = vm_ca * swath_info["freq_zero"] * k_psi * 2 / _constants.speed_of_light
        k_a = npp.polyval(tau - az_rate_t0[az_rate_poly_ind], az_rate_poly_coefs)
        k_t = (k_a * k_s) / (k_a - k_s)
        f_eta_c = npp.polyval(tau - dc_t0[dc_poly_ind], dc_poly_coefs)
        eta = (cols - scp_pixel[1]) * swath_info["ss_zd_s"]
        eta_c = -f_eta_c / k_a  # Beam center crossing time (TimeCOA)
        eta_ref = eta_c - eta_c[0]
        eta_2d, eta_ref_2d = np.meshgrid(eta, eta_ref)
        eta_arg = eta_2d - eta_ref_2d
        deramp_phase = 0.5 * k_t[:, np.newaxis] * eta_arg * eta_arg
        demod_phase = eta_arg * f_eta_c[:, np.newaxis]
        total_phase = deramp_phase + demod_phase

        phase = utils.polyfit2d_tol(
            coords_rg_2d.flatten(),
            coords_az_2d.flatten(),
            total_phase.flatten(),
            poly_order,
            poly_order + 1,
            1e-2,
        )

        # DeltaKCOAPoly is derivative of phase in azimuth/Col direction
        burst_info["col_deltak_coa_poly"] = npp.polyder(phase, axis=1)

        # derive the DopCentroidPoly directly
        burst_info["doppler_centroid_poly_coefs"] = (
            burst_info["col_deltak_coa_poly"] * col_ss / swath_info["ss_zd_s"]
        )

        # complete deriving the TimeCOAPoly, which depends on the DOPCentroidPoly
        time_ca_sampled = npp.polyval(coords_az_2d, time_ca_poly_coefs)
        doppler_rate_sampled = npp.polyval(coords_rg_2d, dr_ca_poly)
        dop_centroid_sampled = npp.polyval2d(
            coords_rg_2d, coords_az_2d, burst_info["doppler_centroid_poly_coefs"]
        )
        time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled

        burst_info["time_coa_poly_coefs"] = utils.polyfit2d_tol(
            coords_rg_2d.flatten(),
            coords_az_2d.flatten(),
            time_coa_sampled.flatten(),
            poly_order,
            poly_order,
            1e-3,
        )

        full_img_verticies = np.array(
            [
                [0, 0],
                [0, num_cols - 1],
                [num_rows - 1, num_cols - 1],
                [num_rows - 1, 0],
            ],
        )
        x_coords = row_ss * (full_img_verticies[:, 0] - (scp_pixel[0] - first_row))
        y_coords = col_ss * (full_img_verticies[:, 1] - (scp_pixel[1] - first_col))

        row_delta_k1, row_delta_k2 = _calc_deltaks(
            x_coords,
            y_coords,
            swath_info["row_deltak_coa_poly"],
            swath_info["row_imp_res_bw"],
            row_ss,
        )
        col_delta_k1, col_delta_k2 = _calc_deltaks(
            x_coords,
            y_coords,
            burst_info["col_deltak_coa_poly"],
            swath_info["col_imp_res_bw"],
            col_ss,
        )
        burst_info["row_delta_k1"] = row_delta_k1
        burst_info["row_delta_k2"] = row_delta_k2
        burst_info["col_delta_k1"] = col_delta_k1
        burst_info["col_delta_k2"] = col_delta_k2

        return time_coa_sampled.min(), time_coa_sampled.max()

    def _update_geo_data_info(swath_info, burst_info):
        scp_drsf = burst_info["drsf_poly_coefs"][0, 0]
        scp_tca = burst_info["time_ca_poly_coefs"][0]
        scp_tcoa = burst_info["time_coa_poly_coefs"][0, 0]
        scp_delta_t_coa = scp_tcoa - scp_tca
        scp_varp_ca_mag = npl.norm(
            npp.polyval(scp_tca, npp.polyder(burst_info["arp_poly_coefs"]))
        )
        scp_rcoa = np.sqrt(
            swath_info["r_ca_scp"] ** 2
            + scp_drsf * scp_varp_ca_mag**2 * scp_delta_t_coa**2
        )
        scp_rratecoa = scp_drsf / scp_rcoa * scp_varp_ca_mag**2 * scp_delta_t_coa
        scp_set = sksicd.projection.ProjectionSetsMono(
            t_COA=np.array([scp_tcoa]),
            ARP_COA=np.array([npp.polyval(scp_tcoa, burst_info["arp_poly_coefs"])]),
            VARP_COA=np.array(
                [npp.polyval(scp_tcoa, npp.polyder(burst_info["arp_poly_coefs"]))]
            ),
            R_COA=np.array([scp_rcoa]),
            Rdot_COA=np.array([scp_rratecoa]),
        )
        scp_ecf = sksicd.projection.r_rdot_to_ground_plane_mono(
            -1,
            scp_set,
            sarkit.wgs84.geodetic_to_cartesian(burst_info["init_scp_llh"]),
            sarkit.wgs84.up(burst_info["init_scp_llh"]),
        )[0]
        scp_llh = sarkit.wgs84.cartesian_to_geodetic(scp_ecf)

        return scp_ecf, scp_llh

    def _calc_grid_unit_vectors(burst_info):
        # Calc Grid unit vectors based on updated RMA, Position, and GeoData
        scp_tca = burst_info["time_ca_poly_coefs"][0]
        scp_ca_pos = npp.polyval(scp_tca, burst_info["arp_poly_coefs"])
        scp_ca_vel = npp.polyval(scp_tca, npp.polyder(burst_info["arp_poly_coefs"]))
        los = burst_info["scp_ecf"] - scp_ca_pos
        row_uvect_ecf = los / npl.norm(los)
        left = np.cross(scp_ca_pos, scp_ca_vel)
        look = np.sign(np.dot(left, row_uvect_ecf))
        spz = -look * np.cross(row_uvect_ecf, scp_ca_vel)
        uspz = spz / npl.norm(spz)

        return row_uvect_ecf, np.cross(uspz, row_uvect_ecf)

    def _compute_burst_id(base_info, swath_info, burst_info):
        """Add `burst_id` to each burst info in `swaths_info` based on DI-MPC-IPFDPM, Issue 2.5, Clause 9.25"""
        t_orb = 12 * 24 * 3600 / 175
        t_pre = {"IW": 2.299849, "EW": 2.299970}[swath_info["mode_id"]]
        t_beam = {"IW": 2.758273, "EW": 3.038376}[swath_info["mode_id"]]

        acq_start_utc = datetime.datetime.fromisoformat(
            base_info["acq_start_str"]
        ).replace(tzinfo=datetime.timezone.utc)
        t_anx = acq_start_utc - datetime.timedelta(
            milliseconds=base_info["start_time_anx"]
        )

        scp_tcoa = burst_info["time_coa_poly_coefs"][0, 0]
        t_b = (
            burst_info["collect_start"] + datetime.timedelta(seconds=scp_tcoa)
        ).replace(tzinfo=datetime.timezone.utc)
        delta_t_b = (t_b - t_anx).total_seconds() + (
            base_info["parameters"]["RELATIVE_ORBIT_NUM"] - 1
        ) * t_orb
        return int(1 + (delta_t_b - t_pre) // t_beam)

    def _finalize_stripmap():
        burst_info = dict()

        scp = _get_scps(swath_info, 1)

        burst_info["scp_ecf"] = scp[0, :]
        burst_info["init_scp_llh"] = sarkit.wgs84.cartesian_to_geodetic(scp[0, :])

        num_rows = swath_info["num_rows"]
        num_cols = swath_info["num_cols"]
        burst_info["valid_data"] = [
            (0, 0),
            (0, num_cols - 1),
            (num_rows - 1, num_cols - 1),
            (num_rows - 1, 0),
        ]

        t_slice = int(_get_slice(product_root_node))
        swath = _get_swath(product_root_node)
        burst_info["core_name"] = (
            f"{swath_info['sensing_start'].strftime('%d%b%YT%H%M%S').upper()}_{product_root_node.find('./adsHeader/missionId').text}{product_root_node.find('./adsHeader/missionDataTakeId').text}_{t_slice:02d}_{swath}_01"
        )

        burst_info["parameters"] = {"BURST": f"{1:02d}"}

        prf = float(
            product_root_node.find(
                "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"
            ).text
        )
        duration = (
            swath_info["sensing_stop"] - swath_info["sensing_start"]
        ).total_seconds()

        burst_info["collect_start"] = swath_info["sensing_start"]
        burst_info["collect_duration"] = duration
        burst_info["ipp_set_tstart"] = 0
        burst_info["ipp_set_tend"] = duration
        burst_info["ipp_set_ippstart"] = 0
        burst_info["ipp_set_ippend"] = round(duration * prf) - 1
        burst_info["tstart_proc"] = 0
        burst_info["tend_proc"] = duration

        burst_info["arp_poly_coefs"] = _compute_arp_poly_coefs(
            product_root_node, swath_info["sensing_start"]
        ).T

        azimuth_time_first_line = dateutil.parser.parse(
            product_root_node.find(
                "./imageAnnotation/imageInformation/productFirstLineUtcTime"
            ).text
        )
        first_line_relative_start = (
            azimuth_time_first_line - swath_info["sensing_start"]
        ).total_seconds()
        _, _ = _calc_rma_and_grid_info(
            swath_info,
            burst_info,
            first_line_relative_start,
            swath_info["sensing_start"],
        )
        burst_info["scp_ecf"], burst_info["scp_llh"] = _update_geo_data_info(
            swath_info, burst_info
        )
        burst_info["row_uvect_ecf"], burst_info["col_uvect_ecf"] = (
            _calc_grid_unit_vectors(burst_info)
        )

        return [burst_info]

    def _finalize_bursts():
        burst_info_list = []

        scps = _get_scps(swath_info, len(burst_list))
        t_slice = int(_get_slice(product_root_node))
        for j, burst in enumerate(burst_list):
            # set preliminary geodata (required for projection)
            burst_info = dict()
            burst_info["scp_ecf"] = scps[j, :]
            burst_info["init_scp_llh"] = sarkit.wgs84.cartesian_to_geodetic(scps[j, :])
            xml_first_cols = np.fromstring(
                burst.find("./firstValidSample").text, sep=" ", dtype=np.int64
            )
            xml_last_cols = np.fromstring(
                burst.find("./lastValidSample").text, sep=" ", dtype=np.int64
            )
            valid = (xml_first_cols >= 0) & (xml_last_cols >= 0)
            valid_cols = np.arange(xml_first_cols.size, dtype=np.int64)[valid]
            first_row = int(np.min(xml_first_cols[valid]))
            last_row = int(np.max(xml_last_cols[valid]))
            first_col = valid_cols[0]
            last_col = valid_cols[-1]
            burst_info["valid_data"] = [
                (first_row, first_col),
                (first_row, last_col),
                (last_row, last_col),
                (last_row, first_col),
            ]

            swath = _get_swath(product_root_node)
            # Use burst index for burst_id to maintain SARPy compatibility
            burst_info["core_name"] = (
                f"{swath_info['sensing_start'].strftime('%d%b%YT%H%M%S').upper()}_{product_root_node.find('./adsHeader/missionId').text}{product_root_node.find('./adsHeader/missionDataTakeId').text}_{t_slice:02d}_{swath}_{j + 1:02d}"
            )
            arp_poly_coefs = _compute_arp_poly_coefs(
                product_root_node, swath_info["sensing_start"]
            )
            burst_info["arp_poly_coefs"] = arp_poly_coefs.T

            az_time = dateutil.parser.parse(burst.find("./azimuthTime").text)
            first_line_relative_start = (
                az_time - swath_info["sensing_start"]
            ).total_seconds()
            early, late = _calc_rma_and_grid_info(
                swath_info,
                burst_info,
                first_line_relative_start,
                swath_info["sensing_start"],
            )
            prf = float(
                product_root_node.find(
                    "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"
                ).text
            )
            burst_info["collect_start"] = swath_info["sensing_start"]
            burst_info["collect_duration"] = (
                swath_info["sensing_stop"] - swath_info["sensing_start"]
            ).total_seconds()
            burst_info["ipp_set_tstart"] = early
            burst_info["ipp_set_tend"] = late
            burst_info["ipp_set_ippstart"] = round(early * prf)
            burst_info["ipp_set_ippend"] = round(late * prf) - 1
            burst_info["tstart_proc"] = early
            burst_info["tend_proc"] = late

            if burst.find("./burstId") is not None:
                burst_id = burst.findtext("./burstId")
            else:
                burst_id = _compute_burst_id(base_info, swath_info, burst_info)

            # Keep BURST as the index to maintain SARPy compatibility
            burst_info["parameters"] = {
                "BURST": f"{j + 1:02d}",
                "BURST_ID": burst_id,
            }

            burst_info["arp_poly_coefs"] = arp_poly_coefs.T

            burst_info["scp_ecf"], burst_info["scp_llh"] = _update_geo_data_info(
                swath_info, burst_info
            )
            burst_info["row_uvect_ecf"], burst_info["col_uvect_ecf"] = (
                _calc_grid_unit_vectors(burst_info)
            )

            burst_info_list.append(burst_info)

        return burst_info_list

    if len(burst_list) > 0:
        return _finalize_bursts()
    else:
        return _finalize_stripmap()


def _calc_radiometric_info(cal_file_name, swath_info, burst_info_list):
    """Compute radiometric polys"""
    cal_root_node = lxml.etree.parse(cal_file_name).getroot()
    cal_vector_list = cal_root_node.findall(
        "./{*}calibrationVectorList/{*}calibrationVector"
    )
    line = np.empty((len(cal_vector_list),), dtype=np.float64)
    pixel, sigma, beta, gamma = [], [], [], []
    for i, cal_vector in enumerate(cal_vector_list):
        line[i] = float(cal_vector.find("./line").text)
        pixel.append(
            np.fromstring(cal_vector.find("./pixel").text, sep=" ", dtype=np.float64)
        )
        sigma.append(
            np.fromstring(
                cal_vector.find("./sigmaNought").text, sep=" ", dtype=np.float64
            )
        )
        beta.append(
            np.fromstring(
                cal_vector.find("./betaNought").text, sep=" ", dtype=np.float64
            )
        )
        gamma.append(
            np.fromstring(cal_vector.find("./gamma").text, sep=" ", dtype=np.float64)
        )

    lines_per_burst = swath_info["num_cols"]
    pixel = np.array(pixel)
    sigma = np.array(sigma)
    beta = np.array(beta)
    gamma = np.array(gamma)
    # adjust sentinel values for sicd convention (square and invert)
    sigma = 1.0 / (sigma * sigma)
    beta = 1.0 / (beta * beta)
    gamma = 1.0 / (gamma * gamma)

    for idx, burst_info in enumerate(burst_info_list):
        burst_first_line = idx * lines_per_burst
        burst_last_line = burst_first_line + lines_per_burst
        valid_lines = (line >= burst_first_line) & (line < burst_last_line)
        valid_count = np.sum(valid_lines)
        if valid_count == 0:
            # this burst contained no useful calibration data
            return

        max_poly_order = 4
        first_row = swath_info["first_row"]
        first_col = swath_info["first_col"]
        scp_row = swath_info["scp_pixel"][0]
        scp_col = swath_info["scp_pixel"][1]
        row_ss = swath_info["row_ss"]
        col_ss = swath_info["col_ss"]
        coords_rg = (pixel[valid_lines] + first_row - scp_row) * row_ss
        coords_az = (
            line[valid_lines] + first_col - scp_col - burst_first_line
        ) * col_ss
        # NB: coords_rg = (valid_count, M) and coords_az = (valid_count, )
        coords_az = np.repeat(coords_az, pixel.shape[1])
        if valid_count > 1:
            coords_az = coords_az.reshape((valid_count, -1))
        max_poly_order_rg = min(max_poly_order, len(coords_rg) - 1)
        max_poly_order_az = min(max_poly_order, len(coords_az) - 1)

        sigma_vals = sigma[valid_lines, :].flatten()
        sigma_tol = np.maximum(np.std(sigma_vals) * 1e-1, np.mean(sigma_vals) * 1e-2)
        burst_info["radiometric"]["sigma_zero_poly_coefs"] = utils.polyfit2d_tol(
            coords_rg.flatten(),
            coords_az.flatten(),
            sigma_vals,
            max_poly_order_rg,
            max_poly_order_az,
            sigma_tol,
        )
        beta_vals = beta[valid_lines, :].flatten()
        beta_tol = np.maximum(np.std(beta_vals) * 1e-1, np.mean(beta_vals) * 1e-2)
        burst_info["radiometric"]["beta_zero_poly_coefs"] = utils.polyfit2d_tol(
            coords_rg.flatten(),
            coords_az.flatten(),
            beta_vals,
            max_poly_order_rg,
            max_poly_order_az,
            beta_tol,
        )
        gamma_vals = gamma[valid_lines, :].flatten()
        gamma_tol = np.maximum(np.std(gamma_vals) * 1e-1, np.mean(gamma_vals) * 1e-2)
        burst_info["radiometric"]["gamma_zero_poly_coefs"] = utils.polyfit2d_tol(
            coords_rg.flatten(),
            coords_az.flatten(),
            gamma_vals,
            max_poly_order_rg,
            max_poly_order_az,
            gamma_tol,
        )

        range_weight_f = azimuth_weight_f = 1.0
        row_wgt_funct = swath_info["row_wgts"]
        var = np.var(row_wgt_funct)
        mean = np.mean(row_wgt_funct)
        range_weight_f += var / (mean * mean)

        col_wgt_funct = swath_info["col_wgts"]
        var = np.var(col_wgt_funct)
        mean = np.mean(col_wgt_funct)
        azimuth_weight_f += var / (mean * mean)
        sp_area = (range_weight_f * azimuth_weight_f) / (
            swath_info["row_imp_res_bw"] * swath_info["col_imp_res_bw"]
        )

        burst_info["radiometric"]["rcs_sf_poly_coefs"] = (
            burst_info["radiometric"]["beta_zero_poly_coefs"] * sp_area
        )

    return


def _calc_noise_level_info(noise_file_name, swath_info, burst_info_list):
    """Compute noise poly"""
    noise_root_node = lxml.etree.parse(noise_file_name).getroot()
    mode_id = swath_info["mode_id"]
    lines_per_burst = swath_info["num_cols"]
    range_size_pixels = swath_info["num_rows"]

    def _extract_vector(stem):
        lines, pixels, noises = [], [], []
        noise_vector_list = noise_root_node.findall(
            f"./{stem:s}VectorList/{stem:s}Vector"
        )
        for i, noise_vector in enumerate(noise_vector_list):
            line = np.fromstring(
                noise_vector.find("./line").text, dtype=np.int64, sep=" "
            )
            # some datasets have noise vectors for negative lines - ignore these
            if np.all(line < 0):
                continue

            pixel_node = noise_vector.find(
                "./pixel"
            )  # does not exist for azimuth noise
            if pixel_node is not None:
                pixel = np.fromstring(pixel_node.text, dtype=np.int64, sep=" ")
            else:
                pixel = None
            noise = np.fromstring(
                noise_vector.find(f"./{stem}Lut").text, dtype=np.float64, sep=" "
            )
            # some datasets do not have any noise data (all 0's) - skipping these will throw things into disarray
            if not np.all(noise == 0):
                # convert noise to dB - what about -inf values?
                noise = 10 * np.log10(noise)
            assert isinstance(noise, np.ndarray)

            # do some validity checks
            if (
                (mode_id == "IW")
                and np.any((line % lines_per_burst) != 0)
                and (i != len(noise_vector_list) - 1)
            ):
                # NB: the final burst has different timing
                raise ValueError(
                    "Noise file should have one lut per burst, but more are present"
                )
            if (pixel is not None) and (pixel[-1] > range_size_pixels):
                raise ValueError("Noise file has more pixels in LUT than range size")

            lines.append(line)
            pixels.append(pixel)
            noises.append(noise)
        return lines, pixels, noises

    # extract noise vectors
    if noise_root_node.find("./noiseVectorList") is not None:
        # probably prior to March 2018
        range_line, range_pixel, range_noise = _extract_vector("noise")
    else:
        # noiseRange and noiseAzimuth fields began in March 2018
        range_line, range_pixel, range_noise = _extract_vector("noiseRange")
    range_line = np.concatenate(range_line, axis=0)

    if noise_root_node.find("./noiseAzimuthVectorList/noiseAzimuthVector") is not None:
        azimuth_line, _, azimuth_noise = _extract_vector("noiseAzimuth")
        azimuth_line = np.concatenate(azimuth_line, axis=0)
    else:
        azimuth_line, azimuth_noise = None, None

    rg_poly_order = min(4, range_pixel[0].size - 1)
    first_row = swath_info["first_row"]
    first_col = swath_info["first_col"]
    scp_row = swath_info["scp_pixel"][0]
    scp_col = swath_info["scp_pixel"][1]
    row_ss = swath_info["row_ss"]
    col_ss = swath_info["col_ss"]
    for idx, burst_info in enumerate(burst_info_list):
        if mode_id[0] == "S":
            # STRIPMAP - all LUTs apply
            az_poly_order = min(4, len(range_line) - 1)
            coords_rg = (range_pixel[0] + first_row - scp_row) * row_ss
            coords_az = (range_line + first_col - scp_col) * col_ss

            coords_az_2d, coords_rg_2d = np.meshgrid(coords_az, coords_rg)

            noise_poly = utils.polyfit2d_tol(
                coords_rg_2d.flatten(),
                coords_az_2d.flatten(),
                np.array(range_noise).flatten(),
                rg_poly_order,
                az_poly_order,
                1e-2,
            )
        else:
            # TOPSAR has single LUT per burst
            # Treat range and azimuth polynomial components as weakly independent
            if idx >= len(range_pixel):
                raise ValueError(
                    f"We have run out of noise information. Current index = {idx}, length of noise array = {len(range_pixel)}."
                )
            rp_array = range_pixel[idx]
            rn_array = range_noise[idx]
            coords_rg = (rp_array + first_row - scp_row) * row_ss

            rg_poly = np.array(npp.polyfit(coords_rg, rn_array, rg_poly_order))
            az_poly = None
            if azimuth_noise is not None:
                line0 = lines_per_burst * idx
                coords_az = (azimuth_line[0] - line0 - scp_col) * col_ss
                valid_lines = (azimuth_line[0] >= line0) & (
                    azimuth_line[0] < line0 + lines_per_burst
                )
                valid_count = np.sum(valid_lines)
                if valid_count > 1:
                    az_poly_order = min(2, valid_count - 1)
                    az_poly = np.array(
                        npp.polyfit(
                            coords_az[valid_lines],
                            azimuth_noise[valid_lines],
                            az_poly_order,
                        )
                    )
            if az_poly is not None:
                noise_poly = np.zeros((rg_poly.size, az_poly.size), dtype=np.float64)
                noise_poly[:, 0] += rg_poly
                noise_poly[0, :] += az_poly
            else:
                noise_poly = np.reshape(rg_poly, (-1, 1))

        burst_info["radiometric"]["noise_level_type"] = "ABSOLUTE"
        burst_info["radiometric"]["noise_poly_coefs"] = noise_poly

    return


def _complete_filename(swath_info, burst_info, filename_template):
    swath = swath_info["parameters"]["SWATH"]
    burst = burst_info["parameters"]["BURST"]
    polarization = swath_info["tx_rcv_polarization_proc"].replace(":", "")
    formatted_name = filename_template.name.format(
        swath=swath, burst=burst, pol=polarization
    )
    final_filename = filename_template.with_name(formatted_name)

    return final_filename


def _create_sicd_xml(base_info, swath_info, burst_info, classification):
    sicd_xml_obj = lxml.etree.Element(
        f"{{{NSMAP['sicd']}}}SICD", nsmap={None: NSMAP["sicd"]}
    )
    sicd_ew = sksicd.ElementWrapper(sicd_xml_obj)

    # Collection Info
    parameter = [
        ("SLICE", swath_info["parameters"]["SLICE"]),
        ("BURST", burst_info["parameters"]["BURST"]),
        ("SWATH", swath_info["parameters"]["SWATH"]),
        ("ORBIT_SOURCE", swath_info["parameters"]["ORBIT_SOURCE"]),
    ]
    if burst_info["parameters"].get("BURST_ID") is not None:
        parameter.append(("BURST_ID", str(burst_info["parameters"]["BURST_ID"])))
    parameter.extend(
        [
            ("MISSION_DATA_TAKE_ID", swath_info["parameters"]["MISSION_DATA_TAKE_ID"]),
            ("RELATIVE_ORBIT_NUM", str(base_info["parameters"]["RELATIVE_ORBIT_NUM"])),
        ]
    )
    sicd_ew["CollectionInfo"] = {
        "CollectorName": swath_info["collector_name"],
        "CoreName": burst_info["core_name"],
        "CollectType": base_info["collect_type"],
        "RadarMode": {
            "ModeType": base_info["mode_type"],
            "ModeID": swath_info["mode_id"],
        },
        "Classification": classification,
        "Parameter": parameter,
    }
    sicd_ew["ImageCreation"] = {
        "Application": base_info["creation_application"],
        "DateTime": base_info["creation_date_time"],
    }
    sicd_ew["ImageData"] = {
        "PixelType": swath_info["pixel_type"],
        "NumRows": swath_info["num_rows"],
        "NumCols": swath_info["num_cols"],
        "FirstRow": swath_info["first_row"],
        "FirstCol": swath_info["first_col"],
        "FullImage": {
            "NumRows": swath_info["num_rows"],
            "NumCols": swath_info["num_cols"],
        },
        "SCPPixel": swath_info["scp_pixel"],
        "ValidData": burst_info["valid_data"],
    }

    sicd_ew["GeoData"] = {
        "EarthModel": "WGS_84",
        "SCP": {
            "ECF": burst_info["scp_ecf"],
            "LLH": burst_info["scp_llh"],
        },
    }

    sicd_ew["Grid"] = {
        "ImagePlane": swath_info["image_plane"],
        "Type": swath_info["grid_type"],
        "TimeCOAPoly": burst_info["time_coa_poly_coefs"],
        "Row": {
            "UVectECF": burst_info["row_uvect_ecf"],
            "SS": swath_info["row_ss"],
            "ImpRespWid": swath_info["row_imp_res_wid"],
            "Sgn": swath_info["row_sgn"],
            "ImpRespBW": swath_info["row_imp_res_bw"],
            "KCtr": swath_info["row_kctr"],
            "DeltaK1": burst_info["row_delta_k1"],
            "DeltaK2": burst_info["row_delta_k2"],
            "DeltaKCOAPoly": swath_info["row_deltak_coa_poly"],
            "WgtType": {
                "WindowName": swath_info["row_window_name"],
            },
            "WgtFunct": swath_info["row_wgts"],
        },
        "Col": {
            "UVectECF": burst_info["col_uvect_ecf"],
            "SS": swath_info["col_ss"],
            "ImpRespWid": swath_info["col_imp_res_wid"],
            "Sgn": swath_info["col_sgn"],
            "ImpRespBW": swath_info["col_imp_res_bw"],
            "KCtr": swath_info["col_kctr"],
            "DeltaK1": burst_info["col_delta_k1"],
            "DeltaK2": burst_info["col_delta_k2"],
            "DeltaKCOAPoly": burst_info["col_deltak_coa_poly"],
            "WgtType": {
                "WindowName": swath_info["col_window_name"],
            },
            "WgtFunct": swath_info["col_wgts"],
        },
    }
    if swath_info["row_params"] is not None:
        sicd_ew["Grid"]["Row"]["WgtType"]["Parameter"] = [
            ("COEFFICIENT", str(swath_info["row_params"]))
        ]
    if swath_info["col_params"] is not None:
        sicd_ew["Grid"]["Col"]["WgtType"]["Parameter"] = [
            ("COEFFICIENT", str(swath_info["col_params"]))
        ]

    sicd_ew["Timeline"] = {
        "CollectStart": burst_info["collect_start"],
        "CollectDuration": burst_info["collect_duration"],
        "IPP": {
            "@size": 1,
            "Set": [
                {
                    "@index": 1,
                    "TStart": burst_info["ipp_set_tstart"],
                    "TEnd": burst_info["ipp_set_tend"],
                    "IPPStart": burst_info["ipp_set_ippstart"],
                    "IPPEnd": burst_info["ipp_set_ippend"],
                    "IPPPoly": swath_info["ipp_poly"],
                }
            ],
        },
    }

    sicd_ew["Position"]["ARPPoly"] = burst_info["arp_poly_coefs"]

    # Radar Collection
    sicd_ew["RadarCollection"] = {
        "TxFrequency": {
            "Min": swath_info["tx_freq"][0],
            "Max": swath_info["tx_freq"][1],
        },
        "Waveform": {
            "@size": len(swath_info["rcv_window_length"]),
            "WFParameters": [
                {
                    "@index": i,
                    "TxPulseLength": swath_info["tx_pulse_length"],
                    "TxRFBandwidth": swath_info["tx_rf_bw"],
                    "TxFreqStart": swath_info["tx_freq_start"],
                    "TxFMRate": swath_info["tx_fm_rate"],
                    "RcvWindowLength": swl,
                    "ADCSampleRate": swath_info["adc_sample_rate"],
                }
                for i, swl in enumerate(swath_info["rcv_window_length"], start=1)
            ],
        },
        "TxPolarization": swath_info["tx_polarization"],
        "RcvChannels": {
            "@size": len(base_info["tx_rcv_polarization"]),
            "ChanParameters": [
                {
                    "@index": i,
                    "TxRcvPolarization": entry,
                }
                for i, entry in enumerate(base_info["tx_rcv_polarization"], start=1)
            ],
        },
    }

    chan_index = None
    for i, pol in enumerate(base_info["tx_rcv_polarization"], start=1):
        if pol == swath_info["tx_rcv_polarization_proc"]:
            chan_index = str(i)

    # Image Formation
    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    sicd_ew["ImageFormation"] = {
        "RcvChanProc": {
            "NumChanProc": 1,
            "PRFScaleFactor": 1,
            "ChanIndex": [chan_index],
        },
        "TxRcvPolarizationProc": swath_info["tx_rcv_polarization_proc"],
        "TStartProc": burst_info["tstart_proc"],
        "TEndProc": burst_info["tend_proc"],
        "TxFrequencyProc": {
            "MinProc": swath_info["tx_freq_proc"][0],
            "MaxProc": swath_info["tx_freq_proc"][1],
        },
        "ImageFormAlgo": swath_info["image_form_algo"],
        "STBeamComp": swath_info["st_beam_comp"],
        "ImageBeamComp": swath_info["image_beam_comp"],
        "AzAutofocus": swath_info["az_autofocus"],
        "RgAutofocus": swath_info["rg_autofocus"],
        "Processing": [
            {
                "Type": f"sarkit-convert {__version__} @ {now}",
                "Applied": True,
            },
        ],
    }

    sicd_ew["RMA"] = {
        "RMAlgoType": swath_info["rm_algo_type"],
        "ImageType": swath_info["image_type"],
        "INCA": {
            "TimeCAPoly": burst_info["time_ca_poly_coefs"],
            "R_CA_SCP": swath_info["r_ca_scp"],
            "FreqZero": swath_info["freq_zero"],
            "DRateSFPoly": burst_info["drsf_poly_coefs"],
            "DopCentroidPoly": burst_info["doppler_centroid_poly_coefs"],
            "DopCentroidCOA": swath_info["dop_centroid_coa"],
        },
    }

    # Add Radiometric after Sentinel baseline processing calibration update on 25 Nov 2015.
    if "radiometric" in burst_info:
        # Radiometric
        sicd_ew["Radiometric"]["NoiseLevel"]["NoiseLevelType"] = burst_info[
            "radiometric"
        ]["noise_level_type"]
        sicd_ew["Radiometric"]["NoiseLevel"]["NoisePoly"] = burst_info["radiometric"][
            "noise_poly_coefs"
        ]
        sicd_ew["Radiometric"]["RCSSFPoly"] = burst_info["radiometric"][
            "rcs_sf_poly_coefs"
        ]
        sicd_ew["Radiometric"]["SigmaZeroSFPoly"] = burst_info["radiometric"][
            "sigma_zero_poly_coefs"
        ]
        sicd_ew["Radiometric"]["BetaZeroSFPoly"] = burst_info["radiometric"][
            "beta_zero_poly_coefs"
        ]
        sicd_ew["Radiometric"]["GammaZeroSFPoly"] = burst_info["radiometric"][
            "gamma_zero_poly_coefs"
        ]

    return sicd_xml_obj


def _update_geo_data(sicd_ew):
    # Update ImageCorners
    num_rows = sicd_ew["ImageData"]["NumRows"]
    num_cols = sicd_ew["ImageData"]["NumCols"]
    row_ss = sicd_ew["Grid"]["Row"]["SS"]
    col_ss = sicd_ew["Grid"]["Col"]["SS"]
    scp_pixel = sicd_ew["ImageData"]["SCPPixel"]
    scp_ecf = sicd_ew["GeoData"]["SCP"]["ECF"]
    image_grid_locations = (
        np.array(
            [
                [0, 0],
                [0, num_cols - 1],
                [num_rows - 1, num_cols - 1],
                [num_rows - 1, 0],
            ]
        )
        - scp_pixel
    ) * [row_ss, col_ss]

    icp_ecef, _, _ = sksicd.image_to_ground_plane(
        sicd_ew.elem.getroottree(),
        image_grid_locations,
        scp_ecf,
        sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
    )
    icp_llh = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
    sicd_ew["GeoData"]["ImageCorners"] = icp_llh[:, :2]
    sicd_ew["GeoData"]["ValidData"] = icp_llh[:, :2]


def _update_rniirs_info(sicd_ew):
    info_density, predicted_rniirs = utils.get_rniirs_estimate(sicd_ew)
    sicd_ew["CollectionInfo"].add(
        "Parameter", ("INFORMATION_DENSITY", f"{info_density:.2G}")
    )
    sicd_ew["CollectionInfo"].add(
        "Parameter", ("PREDICTED_RNIIRS", f"{predicted_rniirs:.1f}")
    )


@contextlib.contextmanager
def uint_tiff():
    """Overwrite SAMPLEFORMAT to force TIFFILE not to upcast from complex int16 to complex64

    Per S1-RS-MDA-52-7441 (Sentinel-1 Product Specification) Table 6-13:
    "Interpretation of pixel format. Set to 5 (complex signed integer, 'int16') for SLC products"
    """
    oldformat = tifffile.SAMPLEFORMAT
    tifffile.SAMPLEFORMAT = lambda _: oldformat.UINT  # treat as UINT
    try:
        yield
    finally:
        tifffile.SAMPLEFORMAT = oldformat


def main(args=None):
    """CLI for converting Sentinel SAFE to SICD"""
    parser = argparse.ArgumentParser(
        description="Converts a Sentinel-1 SAFE folder into SICD.",
    )
    parser.add_argument(
        "safe_product_folder",
        type=pathlib.Path,
        help="path of the input SAFE product folder",
    )
    parser.add_argument(
        "classification",
        type=str,
        help="content of the /SICD/CollectionInfo/Classification node in the SICD XML",
    )
    parser.add_argument(
        "output_sicd_file",
        type=pathlib.Path,
        help="path of the output SICD file. The strings '{swath}', '{burst}', '{pol}' will be replaced as appropriate for multiple images",
    )
    config = parser.parse_args(args)

    manifest_filename = config.safe_product_folder / "manifest.safe"

    manifest_root = lxml.etree.parse(manifest_filename).getroot()
    base_info = _collect_base_info(manifest_root)
    files = _get_file_sets(config.safe_product_folder, manifest_root)

    used_filenames = set()
    for entry in files:
        product_root_node = lxml.etree.parse(entry["product"]).getroot()
        swath_info = _collect_swath_info(product_root_node)
        burst_info_list = _collect_burst_info(product_root_node, base_info, swath_info)
        if base_info["creation_date_time"].date() >= np.datetime64("2015-11-25"):
            [burst_info.update({"radiometric": {}}) for burst_info in burst_info_list]
            _calc_radiometric_info(entry["calibration"], swath_info, burst_info_list)
            _calc_noise_level_info(entry["noise"], swath_info, burst_info_list)

        # Grab the data and write the files
        # Overwrite SAMPLEFORMAT to force TIFFILE not to upcast from complex int16 to complex64
        # Treat as uint32 and handle the conversion later
        assert swath_info["pixel_type"] != "RE16I_IM16I", (
            "pixel handling assumes 'RE16I_IM16I'"
        )
        with uint_tiff(), tifffile.TiffFile(entry["data"]) as tif:
            image = tif.asarray().T
            image_width = tif.pages[0].tags.values()[0].value
        begin_col = 0
        for burst_info in burst_info_list:
            sicd_xml_obj = _create_sicd_xml(
                base_info, swath_info, burst_info, config.classification.upper()
            )
            sicd_ew = sksicd.ElementWrapper(sicd_xml_obj)
            # Add SCPCOA node
            sicd_ew["SCPCOA"] = sksicd.compute_scp_coa(sicd_xml_obj.getroottree())
            # Update ImageCorners and ValidData
            _update_geo_data(sicd_ew)

            # RNIIRS calcs require radiometric info
            if "radiometric" in burst_info:
                _update_rniirs_info(sicd_ew)

            # Check for XML consistency
            sicd_con = sarkit.verification.SicdConsistency(sicd_ew.elem)
            sicd_con.check()
            sicd_con.print_result(fail_detail=True)

            end_col = begin_col + sicd_ew["ImageData"]["NumCols"]
            subset = (slice(0, image_width, 1), slice(begin_col, end_col, 1))
            begin_col = end_col
            image_subset = np.ascontiguousarray(image[subset])
            pixel_type = swath_info["pixel_type"]
            view_dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"].newbyteorder(
                image_subset.dtype.byteorder
            )
            complex_data = image_subset.view(view_dtype)

            metadata = sksicd.NitfMetadata(
                xmltree=sicd_ew.elem.getroottree(),
                file_header_part={
                    "ostaid": "ESA",
                    "ftitle": sicd_ew["CollectionInfo"]["CoreName"],
                    "security": {
                        "clas": config.classification[0].upper(),
                        "clsy": "US",
                    },
                },
                im_subheader_part={
                    "iid2": sicd_ew["CollectionInfo"]["CoreName"],
                    "security": {
                        "clas": config.classification[0].upper(),
                        "clsy": "US",
                    },
                    "isorce": sicd_ew["CollectionInfo"]["CollectorName"],
                },
                de_subheader_part={
                    "security": {
                        "clas": config.classification[0].upper(),
                        "clsy": "US",
                    },
                },
            )

            output_filename = _complete_filename(
                swath_info,
                burst_info,
                config.output_sicd_file,
            )

            if output_filename in used_filenames:
                raise ValueError(
                    "Output filename does not include necessary swath, burst and/or polarization slug"
                )

            used_filenames.add(output_filename)

            with output_filename.open("wb") as f:
                with sksicd.NitfWriter(f, metadata) as writer:
                    writer.write_image(complex_data)


if __name__ == "__main__":
    main()
