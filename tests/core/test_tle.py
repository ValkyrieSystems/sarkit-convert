import pytest

import sarkit_convert.tle

TLE_LINES = (
    "ISS (ZARYA)             ",
    "1 25544U 98067A   25348.86277765  .00007299  00000+0  13766-3 0  9992",
    "2 25544  51.6306 131.9662 0003207 249.8734 110.1909 15.49595360543189",
)


def test_iss():
    tle_vals = sarkit_convert.tle.parse_tle(TLE_LINES)[0]
    assert tle_vals["inclination"] == 51.6306
    height = sarkit_convert.tle.orbital_height_from_orbits_per_day(
        tle_vals["mean motion"]
    )
    assert height == pytest.approx(418e3, abs=100)
