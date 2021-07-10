import unittest

import numpy as np

# noinspection PyUnresolvedReferences
from src import vizutils


class TestDataPointsGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dpgen = vizutils.DataPointsGenerator()

    def test_assign_random_state(self):
        with self.assertRaises(AttributeError):
            self.dpgen.__setattr__('__random_state', 27)

    def test_delete_random_state(self):
        with self.assertRaises(AttributeError):
            self.dpgen.__delattr__('__random_state')

    def test_set_random_state(self):
        with self.assertRaises(ValueError):
            self.dpgen.set_random_state(-3)
        self.dpgen.set_random_state(27)
        self.assertEqual(self.dpgen.random_state, 27)
        self.dpgen.set_random_state(0)

    def test_random_state(self):
        self.assertEqual(self.dpgen.random_state, 0)

    def test_gen_float(self):
        self.assertEqual(self.dpgen.random_state, 0)
        float1 = self.dpgen.gen_float()
        float2 = self.dpgen.gen_float(numrange=(2, 3))
        self.assertIsInstance(float1, float)
        self.assertIsInstance(float2, float)
        self.assertGreaterEqual(float1, 0.0)
        self.assertLess(float1, 1.0)
        self.assertGreaterEqual(float2, 2.0)
        self.assertLess(float2, 3.0)

    def test_gen_normal_1d(self):
        arr1 = self.dpgen.gen_normal_1D()
        arr2 = self.dpgen.gen_normal_1D(no_of_points=20)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertEqual(arr1.shape, (1,))
        self.assertEqual(arr2.shape, (20,))

    def test_gen_normal_2d(self):
        arr1 = self.dpgen.gen_normal_2D()
        arr2 = self.dpgen.gen_normal_2D(no_of_points=20)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertEqual(arr1.shape, (1, 2))
        self.assertEqual(arr2.shape, (20, 2))

    def test_gen_normal_3d(self):
        arr1 = self.dpgen.gen_normal_3D()
        arr2 = self.dpgen.gen_normal_3D(no_of_points=20)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertEqual(arr1.shape, (1, 3))
        self.assertEqual(arr2.shape, (20, 3))

    def test_gen_linear1d(self):
        arr1 = self.dpgen.gen_linear1D()
        arr2 = self.dpgen.gen_linear1D(no_of_points=20)
        arr3 = self.dpgen.gen_linear1D(no_of_points=20, is_increasing=False)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertIsInstance(arr3, np.ndarray)
        self.assertEqual(arr1.shape, (1,))
        self.assertEqual(arr2.shape, (20,))
        self.assertEqual(arr3.shape, (20,))
        self.assertGreater(arr2[-1], arr2[0])
        self.assertGreater(arr3[0], arr3[-1])

    def test_gen_linear2d(self):
        arr1 = self.dpgen.gen_linear2D()
        arr2 = self.dpgen.gen_linear2D(no_of_points=20)
        arr3 = self.dpgen.gen_linear2D(no_of_points=20, is_increasing=False)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertIsInstance(arr3, np.ndarray)
        self.assertEqual(arr1.shape, (1, 2))
        self.assertEqual(arr2.shape, (20, 2))
        self.assertEqual(arr3.shape, (20, 2))
        self.assertGreater(arr2[-1][-1], arr2[0][-1])
        self.assertGreater(arr3[0][-1], arr3[-1][-1])

    def test_gen_linear3d(self):
        arr1 = self.dpgen.gen_linear3D()
        arr2 = self.dpgen.gen_linear3D(no_of_points=20)
        arr3 = self.dpgen.gen_linear3D(no_of_points=20, is_increasing=False)
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)
        self.assertIsInstance(arr3, np.ndarray)
        self.assertEqual(arr1.shape, (1, 3))
        self.assertEqual(arr2.shape, (20, 3))
        self.assertEqual(arr3.shape, (20, 3))
        self.assertGreater(arr2[-1][-1], arr2[0][-1])
        self.assertGreater(arr3[0][-1], arr3[-1][-1])

    def test_gen_line(self):
        line1 = self.dpgen.gen_line(1, 0)
        line2 = self.dpgen.gen_line(1, 0, no_points=20)
        self.assertIsInstance(line1, np.ndarray)
        self.assertIsInstance(line2, np.ndarray)
        self.assertEqual(line1.shape, (10, 2))
        self.assertEqual(line2.shape, (20, 2))
        self.assertEqual(line1[-1][-1], line1[-1][0])
        self.assertEqual(line1[0][-1], line1[0][0])
        self.assertEqual(line2[-1][-1], line2[-1][0])
        self.assertEqual(line2[0][-1], line2[0][0])

    def test_gen_plane(self):
        plane1 = self.dpgen.gen_plane(1, 1, 0)
        plane2 = self.dpgen.gen_plane(1, 1, 0, no_points=20)
        plane3 = self.dpgen.gen_plane(1, 1, 0, mesh=True)
        plane4 = self.dpgen.gen_plane(1, 1, 0, no_points=20, mesh=True)
        self.assertIsInstance(plane1, np.ndarray)
        self.assertIsInstance(plane2, np.ndarray)
        self.assertIsInstance(plane3, np.ndarray)
        self.assertIsInstance(plane4, np.ndarray)
        self.assertEqual(plane1.shape, (10, 3))
        self.assertEqual(plane2.shape, (20, 3))
        self.assertEqual(plane3.shape, (3, 10, 10))
        self.assertEqual(plane4.shape, (3, 20, 20))
        self.assertEqual(np.equal((plane1[0] + plane1[1]), plane1[2]).all(), True)
        self.assertEqual(np.equal((plane2[0] + plane2[1]), plane2[2]).all(), True)
        self.assertEqual(np.equal((plane3[0] + plane3[1]), plane3[2]).all(), True)
        self.assertEqual(np.equal((plane4[0] + plane4[1]), plane4[2]).all(), True)

    def test_gen_line_given_x(self):
        x_vals = np.linspace(1, 10, 10)
        line = self.dpgen.gen_line_given_x(x_vals, 1, 0)
        self.assertIsInstance(line, np.ndarray)
        self.assertEqual(line.shape, (10, 2))
        self.assertEqual(list(zip(*line))[0], list(zip(*line))[1])

    def test_gen_plane_given_x(self):
        x1_vals = np.linspace(1, 10, 10)
        x2_vals = np.linspace(1, 10, 10)
        plane1 = self.dpgen.gen_plane_given_x(x1_vals, x2_vals, 1, 1, 0)
        plane2 = self.dpgen.gen_plane_given_x(x1_vals, x2_vals, 1, 1, 0, mesh=True)
        self.assertIsInstance(plane1, np.ndarray)
        self.assertIsInstance(plane2, np.ndarray)
        self.assertEqual(plane1.shape, (10, 3))
        self.assertEqual(plane2.shape, (3, 10, 10))
        self.assertEqual(np.equal((plane1[0] + plane1[1]), plane1[2]).all(), True)
        self.assertEqual(np.equal((plane2[0] + plane2[1]), plane2[2]).all(), True)


if __name__ == '__main__':
    unittest.main()
