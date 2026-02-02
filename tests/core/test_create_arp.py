import pathlib

import lxml
import numpy as np
import pytest
import sarkit.sicd as sksicd
import sarkit.wgs84

from sarkit_convert import create_arp_poly as cap

sicd_xml_filenames = [
    "S1A_IW_RAW__0SDV_20250602T135218_20250602T135250_059468_0761DC_3FC1_Channel_IW2_151224.sicd.xml",
    "S1A_IW_RAW__0SDV_20250619T232444_20250619T232516_059722_076A87_E109_Channel_IW1_320291.sicd.xml",
    "S1B_IW_RAW__0SDV_20210709T233156_20210709T233229_027724_034EFB_2B2D_Channel_IW3_101200.sicd.xml",
    "S1B_IW_RAW__0SDV_20210711T135126_20210711T135158_027747_034FBD_708E_Channel_IW2_151224.sicd.xml",
    "S1C_IW_RAW__0SDV_20250527T014921_20250527T014954_002510_0053AA_6A5F_Channel_IW2_135523.sicd.xml",
    "S1C_IW_RAW__0SDV_20250527T135101_20250527T135133_002517_0053E1_EDEB_Channel_IW2_151224.sicd.xml",
    "S1C_IW_RAW__0SDV_20250608T014922_20250608T014954_002685_0058B2_FC36_Channel_IW2_135523.sicd.xml",
    "S1C_IW_RAW__0SDV_20250608T135101_20250608T135134_002692_0058EA_1F12_Channel_IW2_151224.sicd.xml",
    "S1C_IW_RAW__0SDV_20250615T052352_20250615T052425_002789_005BB9_CA6D_Channel_IW2_359460.sicd.xml",
    "S1C_IW_RAW__0SDV_20250618T233138_20250618T233210_002844_005D47_D58C_Channel_IW3_101200.sicd.xml",
]


@pytest.mark.parametrize("quadrant", [0, 1, 2, 3])
@pytest.mark.parametrize("hemisphere", [-1, 1])
@pytest.mark.parametrize("zsgn", [-1, +1])
@pytest.mark.parametrize("inclination", [10, 45, 60, 80, 95, 150, 170])
def test_eci_uv(quadrant, hemisphere, zsgn, inclination):
    longitude = (quadrant + 0.5) * 90.0
    max_lat = inclination if inclination < 90 else 180 - inclination
    latitude = hemisphere * max_lat / 1.5
    altitude = 1e6
    pos = sarkit.wgs84.geodetic_to_cartesian([latitude, longitude, altitude])
    eci_vel = cap.eci_uv(pos, zsgn, inclination)
    upos = pos / np.linalg.norm(pos)
    assert np.dot(eci_vel, upos) == pytest.approx(0.0)
    assert np.cross(upos, eci_vel)[2] == pytest.approx(np.cos(np.deg2rad(inclination)))


@pytest.mark.parametrize("sicd_xml_filename", sicd_xml_filenames)
def test_create_arp_poly(sicd_xml_filename):
    sentinel_orbit_height_m = 702900
    sentinel_orbit_inclination_deg = 98.18235

    sicd_xml_path = (
        pathlib.Path(__file__).parent / "data/sentinel_sicd_xml" / sicd_xml_filename
    )

    xmlroot = lxml.etree.parse(sicd_xml_path).getroot()
    sicdew = sksicd.ElementWrapper(xmlroot)

    arp_pos = sicdew["SCPCOA"]["ARPPos"]
    arp_vel = sicdew["SCPCOA"]["ARPVel"]
    arp_acc = sicdew["SCPCOA"]["ARPAcc"]
    scp_pos_ecef = sicdew["GeoData"]["SCP"]["ECF"]
    incidence_deg = 90.0 - sicdew["SCPCOA"]["GrazeAng"]
    look_direction = {"L": +1, "R": -1}[sicdew["SCPCOA"]["SideOfTrack"]]
    orbit_direction = np.sign(arp_vel[2])
    angle_to_north_deg = sicdew["SCPCOA"]["AzimAng"]
    doppler_cone_angle_deg = sicdew["SCPCOA"]["DopplerConeAng"]

    # ARP from subsets of metadata
    atn_arp_poly = cap.create_arp_poly(
        scp_pos_ecef,
        incidence_deg,
        look_direction,
        sentinel_orbit_height_m,
        sentinel_orbit_inclination_deg,
        angle_to_north_deg=angle_to_north_deg,
    )

    oddca_arp_poly = cap.create_arp_poly(
        scp_pos_ecef,
        incidence_deg,
        look_direction,
        sentinel_orbit_height_m,
        sentinel_orbit_inclination_deg,
        orbit_direction=orbit_direction,
        doppler_cone_angle_deg=doppler_cone_angle_deg,
    )

    def ang(vec1, vec2):
        return np.rad2deg(
            np.arccos(np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))
        )

    def mag(vec1, vec2):
        return np.abs(np.linalg.norm(vec1) - np.linalg.norm(vec2))

    def check_errs(arp_pos, arp_vel, arp_acc, arp_poly):
        pos_err_m = np.linalg.norm(arp_pos - arp_poly[..., 0])
        vel_err_mag = mag(arp_vel, arp_poly[..., 1])
        vel_err_deg = ang(arp_vel, arp_poly[..., 1])
        acc_err_mag = mag(arp_acc, arp_poly[..., 2] * 2)
        acc_err_deg = ang(arp_acc, arp_poly[..., 2] * 2)

        assert pos_err_m < 10e3
        assert vel_err_deg < 0.1
        assert vel_err_mag < 5.0
        assert acc_err_deg < 0.1
        assert acc_err_mag < 0.02

    check_errs(arp_pos, arp_vel, arp_acc, atn_arp_poly)
    check_errs(arp_pos, arp_vel, arp_acc, oddca_arp_poly)


@pytest.mark.parametrize("quadrant", [0, 1, 2, 3])
@pytest.mark.parametrize("hemisphere", [-1, 1])
@pytest.mark.parametrize("inclination", [10, 45, 60, 80, 95, 150, 170])
def test_unsolveable_setup(quadrant, hemisphere, inclination):
    longitude = (quadrant + 0.5) * 90.0
    max_lat = inclination if inclination < 90 else 180 - inclination
    latitude = hemisphere * (max_lat + 1)
    altitude = 1e6
    pos = sarkit.wgs84.geodetic_to_cartesian([latitude, longitude, altitude])
    with pytest.raises(ValueError, match="is inconsistent with an orbital inclination"):
        cap.eci_uv(pos, 1, inclination)


def test_insufficient_arguments():
    with pytest.raises(ValueError, match="must be specified"):
        cap.create_arp_poly([6.378e6, 0, 0], 45, 1, 1e6, 45)
    with pytest.raises(ValueError, match="must be specified"):
        cap.create_arp_poly([6.378e6, 0, 0], 45, 1, 1e6, 45, orbit_direction=1)
    with pytest.raises(ValueError, match="must be specified"):
        cap.create_arp_poly([6.378e6, 0, 0], 45, 1, 1e6, 45, doppler_cone_angle_deg=90)
