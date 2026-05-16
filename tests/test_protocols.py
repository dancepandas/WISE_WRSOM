"""协议接口测试。"""
import numpy as np

from wise_wrsom.decision.topsis import TOPSISDecision
from wise_wrsom.objectives import WaterDurationObjective
from wise_wrsom.optimizers.smpso import SMPSOOptimizer
from wise_wrsom.protocols.decision import DecisionModel
from wise_wrsom.protocols.objective import ObjectiveFunction
from wise_wrsom.protocols.optimizer import Optimizer
from wise_wrsom.protocols.routing import RoutingModel, RoutingResult, RiverParams
from wise_wrsom.routing.muskingum import MuskingumRouter


class TestProtocolConformance:
    """测试各实现类是否符合协议定义。"""

    def test_muskingum_router_implements_routing_model(self):
        router = MuskingumRouter()
        assert isinstance(router, RoutingModel)

    def test_water_duration_implements_objective(self):
        obj = WaterDurationObjective()
        assert isinstance(obj, ObjectiveFunction)

    def test_smpso_implements_optimizer(self):
        opt = SMPSOOptimizer()
        assert isinstance(opt, Optimizer)

    def test_topsis_implements_decision(self):
        d = TOPSISDecision()
        assert isinstance(d, DecisionModel)


class TestRoutingResult:
    def test_creation(self):
        r = RoutingResult(
            water_duration=np.array([1.0]),
            outflow=np.array([2.0]),
            surface_area=np.array([3.0]),
            infiltration=np.array([4.0]),
        )
        assert r.water_duration[0] == 1.0
        assert len(r.downstream_flows) == 0

    def test_with_downstream_flows(self):
        r = RoutingResult(
            water_duration=np.array([1.0]),
            outflow=np.array([2.0]),
            surface_area=np.array([3.0]),
            infiltration=np.array([4.0]),
            downstream_flows=[np.zeros((9, 1, 80))],
        )
        assert len(r.downstream_flows) == 1


class TestRiverParams:
    def test_creation(self, config):
        p = RiverParams.from_muskingum(config.muskingum)
        assert len(p.k) == 9
