import argparse
import json
import math
import pathlib
import string

import sarkit.wgs84


def _checksum_val(c):
    """The value to add to the checksum for a character"""
    if c in string.digits:
        return int(c)
    if c == "-":
        return 1
    return 0


def parse_tle(lines, lineno=1):
    """Parse a TLE into a dictionary.

    Args
    ----
    lines: list of str
        The lines of the TLE (additional lines allowed)
    lineno: int (optional)
        The line number of the first line for error reporting

    Returns
    -------
    tle: dict
        dict of string to string of TLE fields
    remaining_lines: list of str
        remaining lines from input
    lineno_out: int
        line number of the first of remaining_lines

    """
    retval = {}
    if lines[0][0] != "1":
        retval["title"] = lines[0]
        lines = lines[1:]
        lineno += 1
    ln, lines = lines[0], lines[1:]
    if ln[0] != "1":
        raise ValueError(f"{lineno}: TLE line 1 must start with 1: {ln}")
    retval["satellite catalog number"] = ln[2:7]
    retval["classification"] = ln[7]
    retval["international designator"] = {
        "year": int(ln[9:11]),
        "launch number": int(ln[11:14]),
        "piece": ln[14:17],
    }
    retval["epoch"] = {"year": int(ln[18:20]), "day": float(ln[20:32])}
    retval["derivative of mean motion"] = float(ln[33:43])
    retval["second derivative of mean motion"] = float(
        ln[44] + "." + ln[45:50]
    ) * 10 ** int(ln[50:52])
    retval["drag term"] = float(ln[53] + "." + ln[54:59]) * 10 ** int(ln[59:61])
    retval["type"] = int(ln[62])
    retval["set number"] = int(ln[64:68])
    retval["checksum1"] = int(ln[68])
    checksum1 = sum(_checksum_val(x) for x in ln[:-1]) % 10
    if checksum1 != retval["checksum1"]:
        raise ValueError(f"{lineno}: checksum does not match")

    ln, lines = lines[0], lines[1:]
    lineno += 1
    if ln[0] != "2":
        raise ValueError(f"{lineno}: TLE line 2 must start with 2: {ln}")
    if ln[2:7] != retval["satellite catalog number"]:
        raise ValueError(f"{lineno}: TLE line 2 catalog number does not match line 1")
    retval["inclination"] = float(ln[8:16])
    retval["right ascension of ascending node"] = float(ln[17:25])
    retval["eccentricity"] = float("." + ln[26:33])
    retval["argument of perigee"] = float(ln[34:42])
    retval["mean anomaly"] = float(ln[43:51])
    retval["mean motion"] = float(ln[52:63])
    retval["revolution number at epoch"] = int(ln[63:68])
    retval["checksum2"] = int(ln[68])

    checksum2 = sum(_checksum_val(x) for x in ln[:-1]) % 10
    if checksum2 != retval["checksum2"]:
        raise ValueError(f"{lineno}: checksum does not match")

    return retval, lines, lineno


def orbital_height_from_orbits_per_day(orbits_per_day):
    n = orbits_per_day * 2 * math.pi / (24 * 60 * 60)
    avg_radius = (sarkit.wgs84.GM / n**2) ** (1 / 3)
    return avg_radius - sarkit.wgs84.SEMI_MAJOR_AXIS


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=pathlib.Path)
    config = parser.parse_args(args)

    tle_lines = config.input_filename.read_text().splitlines()
    lineno = 1
    while tle_lines:
        tle_vals, tle_lines, lineno = parse_tle(tle_lines, lineno)
        print(json.dumps(tle_vals, indent=4))
        print(f"{orbital_height_from_orbits_per_day(tle_vals['mean motion'])=}")


if __name__ == "__main__":
    main()
