"""CLI 测试。"""
import pytest
from click.testing import CliRunner

from wise_wrsom.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "WISE-WRSOM" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2.0.0" in result.output

    def test_list_objectives(self, runner):
        result = runner.invoke(cli, ["list-objectives"])
        assert result.exit_code == 0
        assert "water_duration" in result.output
        assert "groundwater_recharge" in result.output

    def test_list_optimizers(self, runner):
        result = runner.invoke(cli, ["list-optimizers"])
        assert result.exit_code == 0
        assert "smpso" in result.output
        assert "nsga3" in result.output
        assert "moead" in result.output

    def test_list_routing(self, runner):
        result = runner.invoke(cli, ["list-routing"])
        assert result.exit_code == 0
        assert "muskingum" in result.output

    def test_init_config(self, runner):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.yaml")
            result = runner.invoke(cli, ["init-config", "-o", path])
            assert result.exit_code == 0
            assert os.path.exists(path)

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--algorithm" in result.output

    def test_optimize_help(self, runner):
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--algorithm" in result.output

    def test_rank_help(self, runner):
        result = runner.invoke(cli, ["rank", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
