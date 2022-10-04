import unittest
import numpy as np
import torch

from fume import calculate_fundamental_matrix, FumeImageTranslation
from torch.autograd import gradcheck

# define two constructed projection matrices
P1 = np.array([[4., 0., 0.5, 50.],
               [0., 4., 0.5, 50.],
               [0., 0., 0.01, 1.]])
P2 = np.array([[-0.5, -0., 4., 50.],
               [-0.5, -4., -0., 50.],
               [-0.01, -0., -0., 1.]])

# define sample points
CORNERS = np.array([[-10, -10, -10],
                    [-10, -10, 10],
                    [-10, 10, -10],
                    [10, -10, -10],
                    [-10, 10, 10],
                    [10, -10, 10],
                    [10, 10, -10],
                    [10, 10, 10],
                    [0, 0, 0],
                    [5, 5, 0]])


def project_points(points: np.ndarray, pmat: np.ndarray):
    assert points.ndim == 2 and points.shape[1] == 3
    S_ijk = np.pad(points, ((0, 0), (0, 1)), constant_values=1)  # make homogenous
    samples_dec_homog = pmat @ S_ijk.T
    samples_dec = samples_dec_homog[:2] / samples_dec_homog[-1]
    return samples_dec


class FundamentalMatrixTest(unittest.TestCase):

    def test_shapes(self):
        with self.assertRaises(AssertionError):
            calculate_fundamental_matrix(np.eye(3, 3), np.eye(1))

    def test_points_on_line(self):
        # calculate the corresponding Fundamental matrix
        F21 = calculate_fundamental_matrix(P_src=P1, P_dst=P2)

        # map a set of 3d points along the two defined projections
        corn_p1 = project_points(CORNERS, P1)
        corn_p2 = project_points(CORNERS, P2)

        # check line-criterion for each pair of projected points and the epipolar line
        for x1, x2 in zip(corn_p1.T, corn_p2.T):
            # compute line coefficients ax + by + c = 0, l2 @ (x, y, 1)^T = 0, l2 = (a, b, c)^T
            l2 = F21 @ np.pad(x1, (0, 1), constant_values=1)

            # convert to direct form y = mx + t --> m = -a/b, t = -c/b
            m, t = -l2[0] / l2[1], -l2[2] / l2[1]

            # test if the point x2 is on the calculated line l2
            self.assertAlmostEqual(x2[1], x2[0] * m + t)


class GradientTest(unittest.TestCase):
    def setUp(self):
        assert torch.cuda.is_available()
        self.F12 = torch.tensor(calculate_fundamental_matrix(P_src=P2, P_dst=P1).reshape((1, 3, 3)), device='cuda')
        self.F21 = torch.tensor(calculate_fundamental_matrix(P_src=P1, P_dst=P2).reshape((1, 3, 3)), device='cuda')

    def test_grad_random(self):
        # random input
        view1 = torch.randn((1, 1, 100, 100), dtype=torch.double, device='cuda', requires_grad=True)

        # view2 = FumeImageTranslation.apply(view1, F12, F21)
        passed = gradcheck(func=FumeImageTranslation.apply,
                           inputs=(view1, self.F21, self.F12))

        self.assertTrue(passed)

    def test_grad_dirac(self):
        # discrete 2d dirac input
        view1 = np.zeros((1, 1, 100, 100), dtype=np.float64)
        view1[0, 0, 40, 50] = 1
        view1 = torch.tensor(view1, dtype=torch.double, device='cuda', requires_grad=True)

        # view2 = FumeImageTranslation.apply(view1, F12, F21)
        passed = gradcheck(func=FumeImageTranslation.apply,
                           inputs=[view1, self.F21, self.F12])

        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
