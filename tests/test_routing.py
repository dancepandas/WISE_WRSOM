"""河道流量计算测试。"""
import numpy as np

from wise_wrsom.protocols.routing import RoutingResult
from wise_wrsom.routing.muskingum import MuskingumRouter


class TestMuskingumRouter:
    def test_implements_protocol(self):
        from wise_wrsom.protocols.routing import RoutingModel
        assert isinstance(MuskingumRouter(), RoutingModel)

    def test_compute_coefficients(self):
        router = MuskingumRouter()
        k = [0, 0, 0, 0, 13, 16, 16, 80, 150]
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0.2]
        t = [24, 24, 24, 24, 24, 24, 24, 24, 24]
        coeff = router._compute_coefficients(k, x, t)
        assert coeff.shape == (9, 3)
        for i in range(9):
            assert np.isclose(coeff[i].sum(), 1.0, atol=1e-10)

    def test_evaluate_quadratic_infiltration(self):
        flow = np.array([[10.0, 20.0, 0.0], [5.0, 0.0, 15.0]])
        cost_coeff = np.array([
            [0.001, 0.002, 0.003],
            [0.01, 0.02, 0.03],
            [0.1, 0.2, 0.3],
        ])
        result = MuskingumRouter._evaluate_quadratic(flow, cost_coeff, 0, clamp_negative=True)
        assert result.shape == flow.shape
        assert np.all(result >= 0)
        assert result[0, 2] == 0
        assert result[1, 1] == 0

    def test_evaluate_quadratic_surface_area(self):
        flow = np.array([[10.0, 20.0, 0.0]])
        surface_coeff = np.array([
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ])
        result = MuskingumRouter._evaluate_quadratic(flow, surface_coeff, 0, clamp_negative=False)
        assert result.shape == flow.shape
        assert result[0, 0] == 0.1 * 100 + 1.0 * 10 + 10.0
        assert result[0, 2] == 0

    def test_load_tributary_data(self, config):
        router = MuskingumRouter()
        path = config.muskingum.tributary_data_path
        trib = router._load_tributary_data(path, 9, 80)
        assert trib.shape[0] == 9

    def test_full_compute(self, sample_population, river_params):
        router = MuskingumRouter()
        result = router.compute(sample_population, river_params)
        assert isinstance(result, RoutingResult)
        assert result.water_duration.shape == (sample_population.shape[0],)
        assert result.outflow.shape == (sample_population.shape[0],)
        assert result.surface_area.shape == (sample_population.shape[0],)
        assert result.infiltration.shape == (sample_population.shape[0],)
