import subprocess


def test_usage():
    retval = subprocess.run(
        ["python", "-m", "sarkit_convert.csk", "--help"], capture_output=True
    )
    assert retval.returncode == 0
    assert "input_h5_file" in retval.stdout.decode()
