import subprocess
import sys

import pytest

import sarkit_convert.iceye


def test_main_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "sarkit_convert.iceye", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "iceye" in result.stdout


def test_main_errors():
    with pytest.raises(SystemExit):
        sarkit_convert.iceye.main()

    with pytest.raises(FileNotFoundError, match="Unable to synchronously open file"):
        sarkit_convert.iceye.main(
            ["/fake/path", "U", "/another/fake/path", "fake_ostaid"]
        )
