"""目标函数测试。"""
import numpy as np

from wise_wrsom.objectives import (
    GroundwaterRechargeObjective,
    OutflowControlObjective,
    SurfaceAreaObjective,
    WaterBalanceObjective,
    WaterDurationObjective,
    create_objectives,
    get_objective,
    list_objectives,
)
from wise_wrsom.protocols.routing import RoutingResult


class TestRegistry:
    def test_list_objectives(self):
        names = list_objectives()
        assert "water_duration" in names
        assert "groundwater_recharge" in names
        assert "surface_area" in names
        assert "outflow_control" in names
        assert len(names) == 5

    def test_get_objective(self):
        cls = get_objective("water_duration")
        obj = cls()
        assert obj.name == "全线通水时长"
        assert obj.direction == "max"

    def test_get_unknown_objective(self):
        import pytest
        with pytest.raises(KeyError):
            get_objective("unknown")

    def test_create_objectives(self):
        objs = create_objectives(["water_duration", "surface_area"])
        assert len(objs) == 2
        assert objs[0].name == "全线通水时长"
        assert objs[1].name == "水面面积"


class TestWaterDurationObjective:
    def test_compute(self, sample_routing_result, sample_population):
        obj = WaterDurationObjective()
        total_water = 150_000_000
        result = obj.compute(sample_routing_result, sample_population, total_water)
        assert result.shape == (len(sample_routing_result.water_duration),)
        assert result.dtype == np.float64

    def test_direction(self):
        obj = WaterDurationObjective()
        assert obj.direction == "max"


class TestGroundwaterRechargeObjective:
    def test_compute(self, sample_routing_result, sample_population):
        obj = GroundwaterRechargeObjective()
        result = obj.compute(sample_routing_result, sample_population, 150_000_000)
        assert result.shape == (len(sample_routing_result.infiltration),)

    def test_direction(self):
        assert GroundwaterRechargeObjective().direction == "max"


class TestSurfaceAreaObjective:
    def test_compute(self, sample_routing_result, sample_population):
        obj = SurfaceAreaObjective()
        result = obj.compute(sample_routing_result, sample_population, 150_000_000)
        assert result.shape == (len(sample_routing_result.surface_area),)

    def test_direction(self):
        assert SurfaceAreaObjective().direction == "max"


class TestOutflowControlObjective:
    def test_compute(self, sample_routing_result, sample_population):
        obj = OutflowControlObjective()
        result = obj.compute(sample_routing_result, sample_population, 150_000_000)
        assert result.shape == (len(sample_routing_result.outflow),)

    def test_direction(self):
        assert OutflowControlObjective().direction == "min"
