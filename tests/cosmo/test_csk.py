import subprocess

import numpy as np
import numpy.polynomial.polynomial as npp
import pytest

import sarkit_convert.csk


def test_usage():
    retval = subprocess.run(
        ["python", "-m", "sarkit_convert.csk", "--help"], capture_output=True
    )
    assert retval.returncode == 0
    assert "input_h5_file" in retval.stdout.decode()


@pytest.fixture
def data():
    order = 5
    rng = np.random.default_rng(seed=482025)
    position_poly = rng.uniform(size=(order + 1, 3))
    time_span = (0, 1)
    ntimes = 101
    times = np.linspace(-10, 10, ntimes)
    positions = npp.polyval(times, position_poly).T
    velocities = npp.polyval(times, npp.polyder(position_poly)).T
    accelerations = npp.polyval(times, npp.polyder(position_poly, 2)).T
    return dict(locals())


def test_fit_state_vectors_nom(data):
    np.testing.assert_allclose(
        data["position_poly"],
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"], data["times"], data["positions"], order=data["order"]
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"][::10],
            data["positions"][::10, :],
            order=data["order"],
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"],
            data["positions"],
            data["velocities"],
            order=data["order"],
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"],
            data["positions"],
            data["velocities"],
            data["accelerations"],
            order=data["order"],
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"],
            data["positions"],
            None,
            data["accelerations"],
            order=data["order"],
        ),
    )


def test_fit_state_vectors_insufficient_states(data):
    dsf = 50
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 1
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf, :],
            order=fail_order,
        )
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 2
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf],
            data["velocities"][::dsf],
            order=fail_order,
        )
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 3
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf],
            data["velocities"][::dsf],
            data["accelerations"][::dsf],
            order=fail_order,
        )
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 2
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        sarkit_convert.csk.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf],
            None,
            data["accelerations"][::dsf],
            order=fail_order,
        )
