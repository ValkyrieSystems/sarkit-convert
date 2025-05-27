import pathlib

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import pytest
import sarkit.sicd as sksicd

import sarkit_convert._utils as utils

NSMAP = {
    "sicd": "urn:SICD:1.4.0",
}


DATAPATH = pathlib.Path(__file__).parents[2] / "data"

good_sicd_xml_path = DATAPATH / "example-sicd-1.3.0.xml"


def test_polyfit2d():
    x, y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-2, 2, 53), indexing="ij")
    rng = np.random.default_rng(seed=482025)
    poly = rng.uniform(-1, 1, size=(3, 4))
    z = npp.polyval2d(x, y, poly)
    coefs = utils.polyfit2d(x.flatten(), y.flatten(), z.flatten(), 2, 3)
    assert np.allclose(coefs, poly)


@pytest.mark.parametrize(
    "x, y, z, error_msg",
    [
        # x not 1D
        (np.array([[1, 2]]), np.array([1, 2]), np.array([1, 2]), "one dimensional"),
        # y not 1D
        (np.array([1, 2]), np.array([[1, 2]]), np.array([1, 2]), "one dimensional"),
        # z is 3D
        (
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([[[1]], [[2]]]),
            "one or two dimensional",
        ),
        # mismatched lengths
        (
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            "same leading dimension",
        ),
    ],
)
def test_polyfit2d_invalid_inputs(x, y, z, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        utils.polyfit2d(x, y, z, 1, 1)


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
        utils.fit_state_vectors(
            data["time_span"], data["times"], data["positions"], order=data["order"]
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        utils.fit_state_vectors(
            data["time_span"],
            data["times"][::10],
            data["positions"][::10, :],
            order=data["order"],
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        utils.fit_state_vectors(
            data["time_span"],
            data["times"],
            data["positions"],
            data["velocities"],
            order=data["order"],
        ),
    )
    np.testing.assert_allclose(
        data["position_poly"],
        utils.fit_state_vectors(
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
        utils.fit_state_vectors(
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
        utils.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf, :],
            order=fail_order,
        )
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 2
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        utils.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf],
            data["velocities"][::dsf],
            order=fail_order,
        )
    with pytest.raises(ValueError, match="Not enough state vectors"):
        ncomp = 3
        fail_order = (1 + (data["ntimes"] - 1) // dsf) * ncomp
        utils.fit_state_vectors(
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
        utils.fit_state_vectors(
            data["time_span"],
            data["times"][::dsf],
            data["positions"][::dsf],
            None,
            data["accelerations"][::dsf],
            order=fail_order,
        )


def test_broadening_from_amp_smoke():
    amps = np.ones(1024)
    broadening = utils.broadening_from_amp(amps)
    assert np.isclose(broadening, 0.8859, rtol=1e-4)


def test_get_rniirs_estimate_smoke():
    sicd_etree = lxml.etree.parse(good_sicd_xml_path)
    xml_helper = sksicd.XmlHelper(sicd_etree)

    inf_density, rniirs = utils.get_rniirs_estimate(xml_helper)
    assert inf_density != 0.0
    assert rniirs != 0.0

    xml_helper.set("./{*}ImageFormation/{*}TxRcvPolarizationProc", "H:V")
    xpol_inf_density, xpol_rniirs = utils.get_rniirs_estimate(xml_helper)
    assert inf_density > xpol_inf_density
    assert rniirs > xpol_rniirs


def test_get_rniirs_estimate_failure():
    sicd_etree = lxml.etree.parse(good_sicd_xml_path)
    xml_helper = sksicd.XmlHelper(sicd_etree)
    xml_helper.set("./{*}Radiometric/{*}NoiseLevel/{*}NoiseLevelType", "NOTABSOLUTE")

    with pytest.raises(
        ValueError,
        match="Radiometric.NoiseLevel.NoiseLevelType is not `ABSOLUTE` so no noise estimate can be derived.",
    ):
        _, _ = utils.get_rniirs_estimate(xml_helper)

    xml_helper.set("./{*}Radiometric/{*}NoiseLevel/{*}NoiseLevelType", "ABSOLUTE")
    sigma_zero_sf_poly_node = xml_helper.element_tree.find(
        "./{*}Radiometric/{*}SigmaZeroSFPoly"
    )
    parent = sigma_zero_sf_poly_node.getparent()
    parent.remove(sigma_zero_sf_poly_node)

    with pytest.raises(
        ValueError,
        match="Radiometric.SigmaZeroSFPoly is not populated, so no sigma0 noise estimate can be derived.",
    ):
        _, _ = utils.get_rniirs_estimate(xml_helper)
