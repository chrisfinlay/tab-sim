import sys
import pytest
from unittest.mock import patch
from pathlib import Path
from tabsim.scripts import sim_vis


def test_sim_vis_help_exists(capsys):
    test_args = ["sim-vis", "--help"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            sim_vis.main()

    assert e.value.code == 0  # argparse exits with code 0 for --help
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
    assert "--help" in out


def test_missing_config_file():
    args = ["sim-vis", "--config", "nonexistent.yaml"]

    with patch.object(sys, "argv", args), pytest.raises(SystemExit):
        sim_vis.main()


def test_simulation_runs_with_config(capsys):
    config_path = (
        Path(__file__).parent.parent / "examples" / "test" / "sim_test_16A.yaml"
    )

    assert config_path.is_file(), f"Missing config file: {config_path}"
    args = ["sim-vis", "--config", str(config_path), "-o"]

    with patch.object(sys, "argv", args):
        sim_vis.main()

    output = capsys.readouterr().out
    assert "Total simulation time" in output
