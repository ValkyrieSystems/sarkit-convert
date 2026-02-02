import subprocess
import sys

import pytest

import sarkit_convert.sentinel


def test_main_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "sarkit_convert.sentinel", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "sentinel" in result.stdout


def test_main_errors():
    with pytest.raises(SystemExit):
        sarkit_convert.sentinel.main()

    with pytest.raises(OSError, match="Error reading file"):
        sarkit_convert.sentinel.main(["/fake/path", "U", "/another/fake/path"])
