"""决策模型测试。"""
import numpy as np

from wise_wrsom.decision.topsis import TOPSISDecision


class TestTOPSISDecision:
    def test_basic_ranking(self, sample_objectives_matrix):
        d = TOPSISDecision()
        g = d.rank(sample_objectives_matrix)
        assert g.shape == (sample_objectives_matrix.shape[0],)
        assert np.all(g >= 0)
        assert np.all(g <= 1)

    def test_with_subjective_weights(self, sample_objectives_matrix):
        d = TOPSISDecision()
        g = d.rank(sample_objectives_matrix, subjective_weights=[0.3, 0.3, 0.2, 0.2])
        assert g.shape == (sample_objectives_matrix.shape[0],)

    def test_without_subjective_weights(self, sample_objectives_matrix):
        d = TOPSISDecision()
        g = d.rank(sample_objectives_matrix, subjective_weights=None)
        assert g.shape == (sample_objectives_matrix.shape[0],)

    def test_ranking_order(self):
        # 构造明确的矩阵：第一个解在所有目标上都最好
        matrix = np.array([
            [10.0, 10.0, 10.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
        ])
        d = TOPSISDecision()
        g = d.rank(matrix, directions=["max", "max", "max"])
        assert g[0] > g[2] > g[1]

    def test_min_direction(self):
        # min 方向目标：值越小越好
        matrix = np.array([
            [100.0, 10.0],  # max目标大, min目标小 → 最好
            [50.0, 50.0],   # 中等
            [10.0, 100.0],  # max目标小, min目标大 → 最差
        ])
        d = TOPSISDecision()
        g = d.rank(matrix, directions=["max", "min"])
        assert g[0] > g[1] > g[2]

    def test_mixed_directions(self):
        matrix = np.array([
            [10.0, 5.0, 100.0],
            [5.0, 10.0, 50.0],
        ])
        d = TOPSISDecision()
        g = d.rank(matrix, directions=["max", "max", "min"])
        assert g.shape == (2,)

    def test_default_directions_all_max(self, sample_objectives_matrix):
        d = TOPSISDecision()
        g = d.rank(sample_objectives_matrix)
        assert g.shape == (sample_objectives_matrix.shape[0],)