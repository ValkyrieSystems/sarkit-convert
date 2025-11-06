import numpy as np
import sarkit.wgs84
import scipy.optimize


def eci_uv(pos, z_sgn, inclination_deg):
    upos = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    east = np.cross([0, 0, 1], upos)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(upos, east)
    cos_gclat = (1 - upos[..., 2] ** 2) ** 0.5
    cos_inc = np.cos(np.deg2rad(inclination_deg))
    if abs(cos_inc) > cos_gclat:
        raise ValueError(
            f"Position {pos} is inconsistent with an orbital"
            f" inclination of {inclination_deg} degrees"
        )
    cos_k = cos_inc / cos_gclat
    sin_k = np.sign(z_sgn) * (1 - cos_k**2) ** 0.5
    return north * sin_k + east * cos_k


def create_inclination_ring(scp_pos_ecef, incidence_deg, orbit_height_m):
    inc_rad = np.deg2rad(incidence_deg)
    scp_pos_geo = sarkit.wgs84.cartesian_to_geodetic(scp_pos_ecef)
    scp_up = sarkit.wgs84.up(scp_pos_geo)
    scp_north = sarkit.wgs84.north(scp_pos_geo)
    scp_east = sarkit.wgs84.east(scp_pos_geo)

    cos_inc = np.cos(inc_rad)
    sin_inc = np.sin(inc_rad)
    to_sat = scp_up * cos_inc + scp_north * sin_inc

    def orbit_error(dist):
        sat_pos = scp_pos_ecef + to_sat * dist
        sat_height = sarkit.wgs84.cartesian_to_geodetic(sat_pos)[2]
        return sat_height - orbit_height_m

    dist_radial = scipy.optimize.brentq(
        orbit_error, orbit_height_m / cos_inc / 2, orbit_height_m / cos_inc
    )
    dist_up = dist_radial * cos_inc
    dist_out = dist_radial * sin_inc
    ring_center = scp_pos_ecef + scp_up * dist_up
    return ring_center, dist_out, scp_east, scp_north


def create_arp_poly(
    scp_pos_ecef,
    incidence_deg,
    look_direction,
    orbit_height_m,
    orbit_inclination_deg,
    *,
    angle_to_north_deg=None,
    orbit_direction=None,
    doppler_cone_angle_deg=None,
):
    """Create an aperture reference position polynomial from a subset of metadata
    and orbital parameters

    Args
    ----
    scp_pos_ecef: ndarray, shape=(3, )
        Scene Center Point position in ecef cartesian coordinates
    incidence_deg: float
        Incidence Angle to SCP during imaging
    look_direction: float
        SICD SideOfTrack conveyed as float (LEFT -> 1, RIGHT -> -1)
    orbit_height_m: float
        Orbital height in meters.  Assumed to be constant
    orbit_inclination_deg: float
        Orbital inclination in degrees
    angle_to_north_deg: float|None
        Imaging angle clockwise from north in degrees.  Equivalent to SICD/SCPCOA/AzimAng
    orbit_direction: float|None
        Indicates whether the arp velocity should be ascending or descending.
        Sign of this value will determine the sign of the velocity z component
    doppler_cone_angle_deg: float|None
        Doppler cone angle at SCP COA in degrees.  Equivalent to SICD/SCPCOA/DopplerConeAng

    Returns
    -------
    ndarray, shape=(3, 3)
        Aperture reference position polynomial in ecef cartesian coordinates.
        The origin of this polynomial is the SCP center of aperture

    Notes
    -----
    Either angle_to_north_deg or both orbit_direction and doppler_cone_angle_deg
    must be specified.
    If all three are present, only angle_to_north_deg will be used as it provides the
    most accurate result.
    """

    ring_center, dist_out, scp_east, scp_north = create_inclination_ring(
        scp_pos_ecef, incidence_deg, orbit_height_m
    )

    gm = (
        3.986004418e14  # https://en.wikipedia.org/wiki/Standard_gravitational_parameter
    )
    omega_3 = sarkit.wgs84.NOMINAL_MEAN_ANGULAR_VELOCITY

    def eciuvel_to_ecefvel(ecefpos, eciuvel):
        eciuvel = eciuvel / np.linalg.norm(eciuvel)
        vmag = (gm / np.linalg.norm(ecefpos, axis=-1)) ** 0.5
        eci_vel = vmag * eciuvel
        ecefvel = eci_vel - np.cross(np.array([0, 0, omega_3]), ecefpos)
        return ecefvel

    if angle_to_north_deg:
        angle_to_north_rad = np.deg2rad(angle_to_north_deg)
        arp_pos = ring_center + dist_out * (
            scp_east * np.sin(angle_to_north_rad)
            + scp_north * np.cos(angle_to_north_rad)
        )
        if angle_to_north_deg % 360 < 180:
            eastness = +1
        else:
            eastness = -1

        orbit_direction = look_direction * eastness
    elif orbit_direction and doppler_cone_angle_deg:
        doppler_cone_angle_rad = np.deg2rad(doppler_cone_angle_deg)

        def dca_err(atn):
            arp_pos = ring_center + dist_out * (
                scp_east * np.sin(atn) + scp_north * np.cos(atn)
            )
            arp_vel_eci = eci_uv(arp_pos, orbit_direction, orbit_inclination_deg)
            arp_vel_ecef = eciuvel_to_ecefvel(arp_pos, arp_vel_eci)
            uvel = arp_vel_ecef / np.linalg.norm(arp_vel_ecef)
            los = scp_pos_ecef - arp_pos
            ulos = los / np.linalg.norm(los)
            dca = np.arccos(np.dot(uvel, ulos))
            return doppler_cone_angle_rad - dca

        eastness = orbit_direction * look_direction
        if eastness > 0:
            min_na = 0
            max_na = np.pi
        else:
            min_na = np.pi
            max_na = 2 * np.pi

        angle_to_north_rad = scipy.optimize.brentq(dca_err, min_na, max_na)

        arp_pos = ring_center + dist_out * (
            scp_east * np.sin(angle_to_north_rad)
            + scp_north * np.cos(angle_to_north_rad)
        )
    else:
        raise ValueError(
            "Either angle_to_north_deg or both orbit_direction"
            " and doppler_cone_angle_deg must be specified."
        )

    arp_vel_eci = eci_uv(arp_pos, orbit_direction, orbit_inclination_deg)
    arp_vel = eciuvel_to_ecefvel(arp_pos, arp_vel_eci)
    arp_acc_mag = gm / np.linalg.norm(arp_pos) ** 2
    arp_acc_dir = -arp_pos / np.linalg.norm(arp_pos)
    arp_acc = arp_acc_dir * arp_acc_mag - np.cross(
        [0, 0, omega_3], 2 * arp_vel + np.cross([0, 0, omega_3], arp_pos)
    )
    return np.stack([arp_pos, arp_vel, arp_acc], axis=-1)
