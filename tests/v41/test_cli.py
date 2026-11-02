import pytest

from anatobind.v41 import cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "anatobind.v41" in capsys.readouterr().out


def test_bad_arguments_are_redacted(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--bogus", "/zz_patient_root/CASE_SECRET_000777"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid_arguments" in err
    assert "CASE_SECRET" not in err and "zz_patient_root" not in err
