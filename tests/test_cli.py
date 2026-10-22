import json
from pathlib import Path

from trainer_rsi.cli import main


def test_no_command_prints_help_and_succeeds(capsys):
    assert main([]) == 0
    assert "Full-pipeline optimizer" in capsys.readouterr().out


def test_facts_lists_every_installed_device(capsys):
    assert main(["facts"]) == 0
    out = capsys.readouterr().out
    assert "nvidia.h800_sxm" in out
    assert "nvidia.a100_sxm_80gb" in out


def test_facts_shows_provenance_for_one_fact(capsys):
    assert main(["facts", "nvidia.rtx_a6000", "throughput.bf16"]) == 0
    out = capsys.readouterr().out
    assert "154.8 TFLOP/s" in out
    assert "derived" in out
    assert "nvidia.com" in out


def test_facts_json_is_machine_readable(capsys):
    assert main(["facts", "nvidia.h800_sxm", "interconnect.nvlink.bandwidth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["facts"]["interconnect.nvlink.bandwidth"]["value"] == 400000000000
    assert payload["facts"]["interconnect.nvlink.bandwidth"]["origin"] == "vendor"


def test_analyze_renders_the_report(bundle_path: Path, capsys):
    assert main(["analyze", str(bundle_path)]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA A100 SXM" in out
    assert "headroom" in out
    assert "communication is not covered" in out


def test_analyze_writes_json_and_keeps_stdout_clean(bundle_path: Path, tmp_path: Path, capsys):
    target = tmp_path / "report.json"
    assert main(["analyze", str(bundle_path), "--json", str(target)]) == 0
    assert capsys.readouterr().out == ""
    report = json.loads(target.read_text())
    assert report["device_id"] == "nvidia.a100_sxm_80gb"
    assert next(finding["waste"] for finding in report["findings"]) == "parallel"


def test_missing_bundle_is_an_error_not_a_crash(tmp_path: Path, capsys):
    assert main(["analyze", str(tmp_path / "absent.bundle.json.gz")]) == 1
    assert "trsi: error:" in capsys.readouterr().err


def test_unknown_device_names_what_is_installed(capsys):
    assert main(["facts", "nvidia.rtx_5090"]) == 1
    assert "installed:" in capsys.readouterr().err
