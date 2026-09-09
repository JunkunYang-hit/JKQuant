from pathlib import Path

import yaml

from jkquant.config import load_config
from jkquant.pipeline import run_daily


def test_demo_pipeline_creates_top_k_csv(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    source["data"]["provider"] = "demo"
    source["data"]["cache_dir"] = str(tmp_path / "cache")
    source["report"]["output_dir"] = str(tmp_path / "reports")
    source["strategy"]["top_k"] = 10
    path = tmp_path / "test-config.yaml"
    path.write_text(yaml.safe_dump(source, allow_unicode=True), encoding="utf-8")
    report, summary = run_daily(load_config(path))
    assert report.exists()
    assert len(report.read_text(encoding="utf-8-sig").splitlines()) == 11
    assert "amount_100m" in report.read_text(encoding="utf-8-sig").splitlines()[0]
    assert summary["universe_count"] == 100
