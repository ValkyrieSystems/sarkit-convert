import subprocess

import pytest

import sarkit_convert.sentinel


def test_main_smoke():
    result = subprocess.run(
        ["python", "-m", "sarkit_convert.sentinel", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage: sentinel.py" in result.stdout


def test_main_errors():
    with pytest.raises(SystemExit):
        sarkit_convert.sentinel.main()

    with pytest.raises(OSError, match="Error reading file '/fake/path/manifest.safe'"):
        sarkit_convert.sentinel.main(["/fake/path", "U", "/another/fake/path"])
