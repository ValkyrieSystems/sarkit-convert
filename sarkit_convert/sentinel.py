"""
=====================
Sentinel SAFE to SICD
=====================

Convert a complex image(s) from the Sentinel SAFE to SICD(s).

"""

import argparse
import pathlib

import dateutil.parser
import lxml.builder
import lxml.etree as et
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
    base_info["creation_site"] = (
        f"{facility.attrib['name']}, {facility.attrib['site']}, {facility.attrib['country']}"
    )

    # Radar Collection
    polarizations = root_node.findall(
        "./metadataSection/metadataObject[@ID='generalProductInformation']/metadataWrap/xmlData/{*}standAloneProductInformation/{*}transmitterReceiverPolarisation",
    )

    base_info["tx_rcv_polarization"] = []
    for pol in polarizations:
        base_info["tx_rcv_polarization"].append(f"{pol.text[0]}:{pol.text[1]}")

    return base_info


def _collect_swath_info(product_root_node):
    swath_info = dict()
    burst_list = product_root_node.findall("./swathTiming/burstList/burst")

    # Collection Info
    swath_info["collector_name"] = product_root_node.find("./adsHeader/missionId").text
    swath_info["mode_id"] = product_root_node.find("./adsHeader/mode").text
    t_slice = _get_slice(product_root_node)
    swath = _get_swath(product_root_node)
    swath_info["parameters"] = {
        "SLICE": t_slice,
        "SWATH": swath,
        "ORBIT_SOURCE": "SLC_INTERNAL",
    }

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
    swath_info["t_start"] = 0
    swath_info["ipp_start"] = 0
    swath_info["ipp_poly"] = (0, swath_info["prf"])

    # Image Formation
    swath_info["tx_rcv_polarization_proc"] = (
        f"{swath_info['tx_polarization']}:{swath_info['rcv_polarization']}"
    )
    swath_info["image_form_algo"] = "RMA"
    swath_info["t_start_proc"] = 0
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
    swath_info["row_window_name"] = range_proc.find("./windowType").text.upper()
    swath_info["row_params"] = range_proc.find("./windowCoefficient").text
    if swath_info["row_window_name"] == "HAMMING":
        swath_info["row_wgts"] = scipy.signal.windows.general_hamming(
            512, float(swath_info["row_params"]), sym=True
        )
    elif swath_info["row_window_name"] == "KAISER":
        swath_info["row_wgts"] = scipy.signal.windows.kaiser(
            512, float(swath_info["row_params"]), sym=True
        )
    else:  # Default to UNIFORM
        swath_info["row_window_name"] = "UNIFORM"
        swath_info["row_params"] = None
        swath_info["row_wgts"] = np.ones(256)

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

    swath_info["col_window_name"] = az_proc.find("./windowType").text.upper()
    swath_info["col_params"] = az_proc.find("./windowCoefficient").text
    if swath_info["col_window_name"] == "HAMMING":
        swath_info["col_wgts"] = scipy.signal.windows.general_hamming(
            512, float(swath_info["col_params"]), sym=True
        )
    elif swath_info["col_window_name"] == "KAISER":
        swath_info["col_wgts"] = scipy.signal.windows.kaiser(
            512, float(swath_info["col_params"]), sym=True
        )
    else:  # Default to UNIFORM
        swath_info["col_window_name"] = "UNIFORM"
        swath_info["col_params"] = None
        swath_info["col_wgts"] = np.ones(256)

    swath_info["col_sgn"] = -1
    swath_info["col_kctr"] = 0.0
    swath_info["col_imp_res_bw"] = dop_bw * swath_info["ss_zd_s"] / swath_info["col_ss"]

    row_broadening_factor = utils.broadening_from_amp(swath_info["row_wgts"])
    col_broadening_factor = utils.broadening_from_amp(swath_info["col_wgts"])
    swath_info["row_imp_res_wid"] = row_broadening_factor / swath_info["row_imp_res_bw"]
    swath_info["col_imp_res_wid"] = col_broadening_factor / swath_info["col_imp_res_bw"]

    return swath_info


def _collect_burst_info(product_root_node, swath_info):
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
        poly_order = 2
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

        phase = utils.polyfit2d(
            coords_rg_2d.flatten(),
            coords_az_2d.flatten(),
            total_phase.flatten(),
            poly_order,
            poly_order,
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

        burst_info["time_coa_poly_coefs"] = utils.polyfit2d(
            coords_rg_2d.flatten(),
            coords_az_2d.flatten(),
            time_coa_sampled.flatten(),
            poly_order,
            poly_order,
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

        start = dateutil.parser.parse(
            product_root_node.find(
                "./generalAnnotation/downlinkInformationList/downlinkInformation/firstLineSensingTime"
            ).text
        )
        stop = dateutil.parser.parse(
            product_root_node.find(
                "./generalAnnotation/downlinkInformationList/downlinkInformation/lastLineSensingTime"
            ).text
        )
        slice = int(_get_slice(product_root_node))
        swath = _get_swath(product_root_node)
        burst_info["core_name"] = (
            f"{start.strftime('%d%b%YT%H%M%S').upper()}_{product_root_node.find('./adsHeader/missionId').text}{product_root_node.find('./adsHeader/missionDataTakeId').text}_{slice:02d}_{swath}_01"
        )

        burst_info["parameters"] = {"BURST": f"{1:d}"}
        prf = float(
            product_root_node.find(
                "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"
            ).text
        )
        duration = (stop - start).total_seconds()

        burst_info["collect_start"] = start
        burst_info["collect_duration"] = duration
        burst_info["ipp_set_tend"] = duration
        burst_info["ipp_set_ippend"] = round(duration * prf) - 1
        burst_info["tend_proc"] = duration

        burst_info["arp_poly_coefs"] = _compute_arp_poly_coefs(
            product_root_node, start
        ).T

        azimuth_time_first_line = dateutil.parser.parse(
            product_root_node.find(
                "./imageAnnotation/imageInformation/productFirstLineUtcTime"
            ).text
        )
        first_line_relative_start = (azimuth_time_first_line - start).total_seconds()
        _, _ = _calc_rma_and_grid_info(
            swath_info, burst_info, first_line_relative_start, start
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

            # This is the first and last zero doppler times of the columns in the burst.
            # Not really CollectStart and CollectDuration in SICD (first last pulse time)
            start = dateutil.parser.parse(burst.find("./azimuthTime").text)
            t_slice = int(_get_slice(product_root_node))
            swath = _get_swath(product_root_node)
            burst_info["core_name"] = (
                f"{start.strftime('%d%b%YT%H%M%S').upper()}_{product_root_node.find('./adsHeader/missionId').text}{product_root_node.find('./adsHeader/missionDataTakeId').text}_{t_slice:02d}_{swath}_{j + 1:02d}"
            )

            burst_info["parameters"] = {"BURST": f"{j + 1:d}"}
            arp_poly_coefs = _compute_arp_poly_coefs(product_root_node, start)
            burst_info["arp_poly_coefs"] = arp_poly_coefs.T
            early, late = _calc_rma_and_grid_info(swath_info, burst_info, 0, start)
            new_start = start + np.timedelta64(np.int64(early * 1e6), "us")
            duration = late - early
            prf = float(
                product_root_node.find(
                    "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"
                ).text
            )
            burst_info["collect_start"] = new_start
            burst_info["collect_duration"] = duration
            burst_info["ipp_set_tend"] = duration
            burst_info["ipp_set_ippend"] = round(duration * prf) - 1
            burst_info["tend_proc"] = duration

            # adjust time offset
            burst_info["time_coa_poly_coefs"][0, 0] -= early
            burst_info["time_ca_poly_coefs"][0] -= early

            arp_poly_coefs = np.array(
                [
                    _shift(arp_poly_coefs[i], t_0=-early)
                    for i in range(len(arp_poly_coefs))
                ]
            )
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


def _calc_radiometric_info(cal_file_name, base_sicd_info, sicds):
    # TODO: Compute radiometric polys

    return


def _calc_noise_level_info(noise_file_name, base_sicd_info, sicds):
    # TODO: Compute noise poly and update XML

    return


def _calc_rniirs_info(sicds):
    # TODO: Add RNIIRS info

    return


def _complete_filename(swath_info, burst_info, filename_template):
    core_name = burst_info["core_name"]
    burst = core_name[-2:]
    swath = swath_info["parameters"]["SWATH"]
    polarization = swath_info["tx_rcv_polarization_proc"].replace(":", "")
    formatted_name = filename_template.name.format(
        swath=swath, burst=burst, pol=polarization
    )
    final_filename = filename_template.with_name(formatted_name)

    return final_filename


def _create_sicd_xml(base_info, swath_info, burst_info, classification):
    em = lxml.builder.ElementMaker(namespace=NSMAP["sicd"], nsmap={None: NSMAP["sicd"]})

    # Collection Info
    collection_info_node = em.CollectionInfo(
        em.CollectorName(swath_info["collector_name"]),
        em.CoreName(burst_info["core_name"]),
        em.CollectType(base_info["collect_type"]),
        em.RadarMode(
            em.ModeType(base_info["mode_type"]),
            em.ModeID(swath_info["mode_id"]),
        ),
        em.Classification(classification),
        em.Parameter({"name": "SLICE"}, swath_info["parameters"]["SLICE"]),
        em.Parameter({"name": "BURST"}, burst_info["parameters"]["BURST"]),
        em.Parameter({"name": "SWATH"}, swath_info["parameters"]["SWATH"]),
        em.Parameter(
            {"name": "ORBIT_SOURCE"}, swath_info["parameters"]["ORBIT_SOURCE"]
        ),
    )

    image_creation_node = em.ImageCreation(
        em.Application(base_info["creation_application"]),
        em.DateTime(base_info["creation_date_time"].strftime("%Y-%m-%dT%H:%M:%SZ")),
        em.Site(base_info["creation_site"]),
        em.Profile(f"sarkit-convert {__version__}"),
    )

    # Image Data
    image_data_node = em.ImageData(
        em.PixelType(swath_info["pixel_type"]),
        em.NumRows(str(swath_info["num_rows"])),
        em.NumCols(str(swath_info["num_cols"])),
        em.FirstRow(str(swath_info["first_row"])),
        em.FirstCol(str(swath_info["first_col"])),
        em.FullImage(
            em.NumRows(str(swath_info["num_rows"])),
            em.NumCols(str(swath_info["num_cols"])),
        ),
        em.SCPPixel(
            em.Row(str(swath_info["scp_pixel"][0])),
            em.Col(str(swath_info["scp_pixel"][1])),
        ),
        em.ValidData(
            {"size": "4"},
            em.Vertex(
                {"index": "1"},
                em.Row(str(burst_info["valid_data"][0][0])),
                em.Col(str(burst_info["valid_data"][0][1])),
            ),
            em.Vertex(
                {"index": "2"},
                em.Row(str(burst_info["valid_data"][1][0])),
                em.Col(str(burst_info["valid_data"][1][1])),
            ),
            em.Vertex(
                {"index": "3"},
                em.Row(str(burst_info["valid_data"][2][0])),
                em.Col(str(burst_info["valid_data"][2][1])),
            ),
            em.Vertex(
                {"index": "4"},
                em.Row(str(burst_info["valid_data"][3][0])),
                em.Col(str(burst_info["valid_data"][3][1])),
            ),
        ),
    )

    def _make_xyz(arr):
        return [em.X(str(arr[0])), em.Y(str(arr[1])), em.Z(str(arr[2]))]

    def __make_llh(arr):
        return [em.Lat(str(arr[0])), em.Lon(str(arr[1])), em.HAE(str(arr[2]))]

    # Geo Data
    geo_data_node = em.GeoData(
        em.EarthModel("WGS_84"),
        em.SCP(
            em.ECF(*_make_xyz(burst_info["scp_ecf"])),
            em.LLH(*__make_llh(burst_info["scp_llh"])),
        ),
        em.ImageCorners(),
        em.ValidData(),
    )

    # Grid
    grid_node = em.Grid(
        em.ImagePlane(swath_info["image_plane"]),
        em.Type(swath_info["grid_type"]),
        em.TimeCOAPoly(),
        em.Row(
            em.UVectECF(*_make_xyz(burst_info["row_uvect_ecf"])),
            em.SS(str(swath_info["row_ss"])),
            em.ImpRespWid(str(swath_info["row_imp_res_wid"])),
            em.Sgn(str(swath_info["row_sgn"])),
            em.ImpRespBW(str(swath_info["row_imp_res_bw"])),
            em.KCtr(str(swath_info["row_kctr"])),
            em.DeltaK1(str(burst_info["row_delta_k1"])),
            em.DeltaK2(str(burst_info["row_delta_k2"])),
            em.DeltaKCOAPoly(),
            em.WgtType(
                em.WindowName(
                    str(swath_info["row_window_name"]),
                ),
                *(
                    [
                        em.Parameter(
                            {"name": "COEFFICIENT"},
                            str(swath_info["row_params"]),
                        )
                    ]
                    if swath_info["row_params"] is not None
                    else []
                ),
            ),
        ),
        em.Col(
            em.UVectECF(*_make_xyz(burst_info["col_uvect_ecf"])),
            em.SS(str(swath_info["col_ss"])),
            em.ImpRespWid(str(swath_info["col_imp_res_wid"])),
            em.Sgn(str(swath_info["col_sgn"])),
            em.ImpRespBW(str(swath_info["col_imp_res_bw"])),
            em.KCtr(str(swath_info["col_kctr"])),
            em.DeltaK1(str(burst_info["col_delta_k1"])),
            em.DeltaK2(str(burst_info["col_delta_k2"])),
            em.DeltaKCOAPoly(),
            em.WgtType(
                em.WindowName(
                    str(swath_info["col_window_name"]),
                ),
                *(
                    [
                        em.Parameter(
                            {"name": "COEFFICIENT"},
                            str(swath_info["col_params"]),
                        )
                    ]
                    if swath_info["col_params"] is not None
                    else []
                ),
            ),
        ),
    )
    sksicd.Poly2dType().set_elem(
        grid_node.find("./{*}TimeCOAPoly"), burst_info["time_coa_poly_coefs"]
    )
    sksicd.Poly2dType().set_elem(
        grid_node.find("./{*}Row/{*}DeltaKCOAPoly"), swath_info["row_deltak_coa_poly"]
    )
    sksicd.Poly2dType().set_elem(
        grid_node.find("./{*}Col/{*}DeltaKCOAPoly"), burst_info["col_deltak_coa_poly"]
    )
    wgtfunc = em.WgtFunct()
    sksicd.TRANSCODERS["Grid/Row/WgtFunct"].set_elem(wgtfunc, swath_info["row_wgts"])
    grid_node.find("./{*}Row").append(wgtfunc)
    wgtfunc = em.WgtFunct()
    sksicd.TRANSCODERS["Grid/Col/WgtFunct"].set_elem(wgtfunc, swath_info["col_wgts"])
    grid_node.find("./{*}Col").append(wgtfunc)

    # Timeline
    timeline_node = em.Timeline(
        em.CollectStart(burst_info["collect_start"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
        em.CollectDuration(str(burst_info["collect_duration"])),
        em.IPP(
            {"size": "1"},
            em.Set(
                {"index": "1"},
                em.TStart(str(swath_info["t_start"])),
                em.TEnd(str(burst_info["ipp_set_tend"])),
                em.IPPStart(str(swath_info["ipp_start"])),
                em.IPPEnd(str(burst_info["ipp_set_ippend"])),
                em.IPPPoly(),
            ),
        ),
    )
    sksicd.PolyType().set_elem(
        timeline_node.find("./{*}IPP/{*}Set/{*}IPPPoly"), swath_info["ipp_poly"]
    )

    # Position
    position_node = em.Position(em.ARPPoly())
    sksicd.XyzPolyType().set_elem(
        position_node.find("./{*}ARPPoly"), burst_info["arp_poly_coefs"]
    )

    # Radar Collection
    radar_collection_node = em.RadarCollection(
        em.TxFrequency(
            em.Min(str(swath_info["tx_freq"][0])),
            em.Max(str(swath_info["tx_freq"][1])),
        ),
        em.Waveform(
            {"size": f"{len(swath_info['rcv_window_length'])}"},
            *[
                em.WFParameters(
                    {"index": str(i)},
                    em.TxPulseLength(str(swath_info["tx_pulse_length"])),
                    em.TxRFBandwidth(str(swath_info["tx_rf_bw"])),
                    em.TxFreqStart(str(swath_info["tx_freq_start"])),
                    em.TxFMRate(str(swath_info["tx_fm_rate"])),
                    em.RcvWindowLength(str(swl)),
                    em.ADCSampleRate(str(swath_info["adc_sample_rate"])),
                )
                for i, swl in enumerate(swath_info["rcv_window_length"], start=1)
            ],
        ),
        em.TxPolarization(swath_info["tx_polarization"]),
        em.RcvChannels(
            {"size": f"{len(base_info['tx_rcv_polarization'])}"},
            *[
                em.ChanParameters(
                    {"index": str(i)},
                    em.TxRcvPolarization(entry),
                )
                for i, entry in enumerate(base_info["tx_rcv_polarization"], start=1)
            ],
        ),
    )

    chan_indices = None
    for i, pol in enumerate(base_info["tx_rcv_polarization"], start=1):
        if pol == swath_info["tx_rcv_polarization_proc"]:
            chan_indices = str(i)

    # Image Formation
    image_formation_node = em.ImageFormation(
        em.RcvChanProc(
            em.NumChanProc("1"),
            em.PRFScaleFactor("1"),
            em.ChanIndex(chan_indices),
        ),
        em.TxRcvPolarizationProc(swath_info["tx_rcv_polarization_proc"]),
        em.TStartProc(str(swath_info["t_start_proc"])),
        em.TEndProc(str(burst_info["tend_proc"])),
        em.TxFrequencyProc(
            em.MinProc(str(swath_info["tx_freq_proc"][0])),
            em.MaxProc(str(swath_info["tx_freq_proc"][1])),
        ),
        em.ImageFormAlgo(swath_info["image_form_algo"]),
        em.STBeamComp(swath_info["st_beam_comp"]),
        em.ImageBeamComp(swath_info["image_beam_comp"]),
        em.AzAutofocus(swath_info["az_autofocus"]),
        em.RgAutofocus(swath_info["rg_autofocus"]),
    )

    # TODO: Radiometric
    # radiometric_node = em.Radiometric(
    #     em.NoiseLevel(
    #         em.NoiseLevelType(""),
    #         em.NoisePoly(),
    #     ),
    #     em.RCSSFPoly(),
    #     em.SigmaZeroSFPoly(),
    #     em.BetaZeroSFPoly(),
    #     em.GammaZeroSFPoly(),
    # )

    rma_node = em.RMA(
        em.RMAlgoType(swath_info["rm_algo_type"]),
        em.ImageType(swath_info["image_type"]),
        em.INCA(
            em.TimeCAPoly(),
            em.R_CA_SCP(str(swath_info["r_ca_scp"])),
            em.FreqZero(str(swath_info["freq_zero"])),
            em.DRateSFPoly(),
            em.DopCentroidPoly(),
            em.DopCentroidCOA(swath_info["dop_centroid_coa"]),
        ),
    )
    sksicd.PolyType().set_elem(
        rma_node.find("./{*}INCA/{*}TimeCAPoly"), burst_info["time_ca_poly_coefs"]
    )
    sksicd.Poly2dType().set_elem(
        rma_node.find("./{*}INCA/{*}DRateSFPoly"), burst_info["drsf_poly_coefs"]
    )
    sksicd.Poly2dType().set_elem(
        rma_node.find("./{*}INCA/{*}DopCentroidPoly"),
        burst_info["doppler_centroid_poly_coefs"],
    )

    sicd_xml_obj = em.SICD(
        collection_info_node,
        image_creation_node,
        image_data_node,
        geo_data_node,
        grid_node,
        timeline_node,
        position_node,
        radar_collection_node,
        image_formation_node,
        # radiometric_node
        rma_node,
    )

    return sicd_xml_obj


def _update_geo_data(xml_helper):
    # Update ImageCorners
    num_rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    num_cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    row_ss = xml_helper.load("./{*}Grid/{*}Row/{*}SS")
    col_ss = xml_helper.load("./{*}Grid/{*}Col/{*}SS")
    scp_pixel = xml_helper.load("./{*}ImageData/{*}SCPPixel")
    scp_ecf = xml_helper.load("./{*}GeoData/{*}SCP/{*}ECF")
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
        xml_helper.element_tree,
        image_grid_locations,
        scp_ecf,
        sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
    )
    icp_llh = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
    xml_helper.set("./{*}GeoData/{*}ImageCorners", icp_llh[:, :2])
    xml_helper.set("./{*}GeoData/{*}ValidData", icp_llh[:, :2])


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
    parser.add_argument(
        "--ostaid",
        type=str,
        help="content of the originating station ID (OSTAID) field of the NITF header",
        default="Unknown",
    )
    config = parser.parse_args(args)

    manifest_filename = config.safe_product_folder / "manifest.safe"

    manifest_root = et.parse(manifest_filename).getroot()
    base_info = _collect_base_info(manifest_root)
    files = _get_file_sets(config.safe_product_folder, manifest_root)

    used_filenames = set()
    for entry in files:
        product_root_node = et.parse(entry["product"]).getroot()
        swath_info = _collect_swath_info(product_root_node)
        burst_info_list = _collect_burst_info(product_root_node, swath_info)
        # Coming soon
        # sicds = _calc_radiometric_info(entry["calibration"], base_sicd_info, burst_sicd_info)
        # sicds = _calc_noise_level_info(entry["noise"], base_sicd_info, burst_sicd_info)
        # sicds = _calc_rniirs_info(burst_sicd_info)

        # Grab the data and write the files
        with tifffile.TiffFile(entry["data"]) as tif:
            image = tif.asarray().T
            image_width = tif.pages[0].tags.values()[0].value
        begin_col = 0
        for burst_info in burst_info_list:
            sicd = _create_sicd_xml(
                base_info, swath_info, burst_info, config.classification.upper()
            )
            scp_coa = sksicd.compute_scp_coa(sicd.getroottree())
            # Add SCPCOA node
            sicd.find("./{*}ImageFormation").addnext(scp_coa)
            xml_helper = sksicd.XmlHelper(et.ElementTree(sicd))
            # Update ImageCorners and ValidData
            _update_geo_data(xml_helper)

            # Check for XML consistency
            sicd_con = sarkit.verification.SicdConsistency(sicd)
            sicd_con.check()
            sicd_con.print_result(fail_detail=True)

            end_col = begin_col + xml_helper.load("{*}ImageData/{*}NumCols")
            subset = (slice(0, image_width, 1), slice(begin_col, end_col, 1))
            begin_col = end_col
            image_subset = image[subset]
            pixel_type = swath_info["pixel_type"]
            view_dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"]
            complex_data = np.empty(image_subset.shape, dtype=view_dtype)
            complex_data["real"] = image_subset.real.astype(np.int16)
            complex_data["imag"] = image_subset.imag.astype(np.int16)

            metadata = sksicd.NitfMetadata(
                xmltree=sicd.getroottree(),
                file_header_part={
                    "ostaid": config.ostaid,
                    "ftitle": xml_helper.load("{*}CollectionInfo/{*}CoreName"),
                    "security": {
                        "clas": config.classification[0].upper(),
                        "clsy": "US",
                    },
                },
                im_subheader_part={
                    "iid2": xml_helper.load("{*}CollectionInfo/{*}CoreName"),
                    "security": {
                        "clas": config.classification[0].upper(),
                        "clsy": "US",
                    },
                    "isorce": xml_helper.load("{*}CollectionInfo/{*}CollectorName"),
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
