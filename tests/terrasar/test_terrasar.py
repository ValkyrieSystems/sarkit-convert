import subprocess
import sys

import pytest

import sarkit_convert.terrasar


def test_main_smoke():
    retval = subprocess.run(
        [sys.executable, "-m", "sarkit_convert.terrasar", "--help"], capture_output=True
    )
    assert retval.returncode == 0
    assert "input_xml_file" in retval.stdout.decode()


def test_main_errors():
    with pytest.raises(SystemExit):
        sarkit_convert.terrasar.main()

    with pytest.raises(ValueError, match="is not a file"):
        sarkit_convert.terrasar.main(["/fake/path", "U", "/yet/another/fake/path"])
