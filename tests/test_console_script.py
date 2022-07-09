import subprocess


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def test_cli_just_vak():
    command = ["vak"]
    out, err, returncode = capture(command)
    assert returncode > 0
    assert out == ''
    assert err.startswith('usage: vak ')


def test_cli_help():
    command = ["vak", "-h"]
    out, err, returncode = capture(command)
    assert returncode == 0
    assert out.startswith('usage: vak ')
    assert err == ''
