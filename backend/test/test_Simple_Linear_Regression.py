from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

# noinspection PyUnresolvedReferences
from src import SimpleLinearRegressionVisualizer


class TestSimpleLinearRegressionVisualizer(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.viz1 = SimpleLinearRegressionVisualizer()
        cls.viz2 = SimpleLinearRegressionVisualizer(randomize=True)
        cls.viz3 = SimpleLinearRegressionVisualizer(random_state=27)
        cls.viz4 = SimpleLinearRegressionVisualizer(is_linearly_increasing=True)
        cls.viz5 = SimpleLinearRegressionVisualizer(no_data_points=30)
        plt.switch_backend('agg')

    def test_wrong_instance(self):
        with self.assertRaises(AttributeError):
            SimpleLinearRegressionVisualizer(randomize=True, random_state=27)

    def test_random_state(self):
        rand_state1 = self.viz1.random_state
        rand_state2 = self.viz2.random_state
        rand_state3 = self.viz3.random_state
        self.assertIsInstance(rand_state1, int)
        self.assertIsInstance(rand_state1, int)
        self.assertNotEqual(rand_state2, 0)
        self.assertEqual(rand_state1, 0)
        self.assertEqual(rand_state3, 27)

    def test_data_points(self):
        dp1 = self.viz1.data_points
        dp2 = self.viz2.data_points
        dp3 = self.viz3.data_points
        dp4 = self.viz4.data_points
        dp5 = self.viz5.data_points
        self.assertIsInstance(dp1, np.ndarray)
        self.assertIsInstance(dp2, np.ndarray)
        self.assertIsInstance(dp3, np.ndarray)
        self.assertIsInstance(dp4, np.ndarray)
        self.assertIsInstance(dp5, np.ndarray)
        self.assertEqual(dp1.shape, (20, 2))
        self.assertEqual(dp2.shape, (20, 2))
        self.assertEqual(dp3.shape, (20, 2))
        self.assertEqual(dp4.shape, (20, 2))
        self.assertEqual(dp5.shape, (30, 2))
        self.assertGreater(dp4[-1][-1], dp4[0][-1])
        self.assertGreater(dp1[0][-1], dp1[-1][-1])

    def test_x_values(self):
        xvals1 = self.viz1.x_values
        xvals2 = self.viz2.x_values
        xvals4 = self.viz4.x_values
        xvals5 = self.viz5.x_values
        self.assertIsInstance(xvals1, np.ndarray)
        self.assertIsInstance(xvals2, np.ndarray)
        self.assertIsInstance(xvals4, np.ndarray)
        self.assertIsInstance(xvals5, np.ndarray)
        self.assertEqual(xvals1.shape, (20,))
        self.assertEqual(xvals2.shape, (20,))
        self.assertEqual(xvals4.shape, (20,))
        self.assertEqual(xvals5.shape, (30,))
        self.assertNotEqual(np.equal(xvals1, xvals2).all(), True)
        self.assertGreater(xvals1[-1], xvals1[0])
        self.assertGreater(xvals4[-1], xvals4[0])

    def test_y_values(self):
        yvals1 = self.viz1.y_values
        yvals2 = self.viz2.y_values
        yvals4 = self.viz4.y_values
        yvals5 = self.viz5.y_values
        self.assertIsInstance(yvals1, np.ndarray)
        self.assertIsInstance(yvals2, np.ndarray)
        self.assertIsInstance(yvals4, np.ndarray)
        self.assertIsInstance(yvals5, np.ndarray)
        self.assertEqual(yvals1.shape, (20,))
        self.assertEqual(yvals2.shape, (20,))
        self.assertEqual(yvals4.shape, (20,))
        self.assertEqual(yvals5.shape, (30,))
        self.assertNotEqual(np.equal(yvals1, yvals2).all(), True)
        self.assertGreater(yvals1[0], yvals1[-1])
        self.assertGreater(yvals4[-1], yvals4[0])

    def test_theta1(self):
        theta1 = self.viz1.theta1
        self.assertIsInstance(theta1, float)
        self.assertGreaterEqual(theta1, 0.0)
        self.assertLess(theta1, 1.0)

    def test_theta0(self):
        theta0 = self.viz1.theta0
        self.assertIsInstance(theta0, float)
        self.assertGreaterEqual(theta0, 0.0)
        self.assertLess(theta0, 1.0)

    def test_learning_rate(self):
        lr = self.viz1.learning_rate
        self.assertIsInstance(lr, float)
        self.assertEqual(lr, 0.001)

    def test_initial_regression_line(self):
        ini_reg_line1 = self.viz1.initial_regression_line
        ini_reg_line5 = self.viz5.initial_regression_line
        self.assertIsInstance(ini_reg_line1, np.ndarray)
        self.assertIsInstance(ini_reg_line5, np.ndarray)
        self.assertEqual(ini_reg_line1.shape, (20, 2))
        self.assertEqual(ini_reg_line5.shape, (30, 2))

    def test_current_regression_line(self):
        curr_reg_line1 = self.viz1.current_regression_line
        curr_reg_line5 = self.viz5.current_regression_line
        self.assertIsInstance(curr_reg_line1, np.ndarray)
        self.assertIsInstance(curr_reg_line5, np.ndarray)
        self.assertEqual(curr_reg_line1.shape, (20, 2))
        self.assertEqual(curr_reg_line5.shape, (30, 2))

    def test_predicted_y_values(self):
        pred_yvals1 = self.viz1.predicted_y_values
        pred_yvals5 = self.viz5.predicted_y_values
        self.assertIsInstance(pred_yvals1, np.ndarray)
        self.assertIsInstance(pred_yvals5, np.ndarray)
        self.assertEqual(pred_yvals1.shape, (20,))
        self.assertEqual(pred_yvals5.shape, (30,))

    def test_cost(self):
        cost = self.viz1.cost
        self.assertIsInstance(cost, float)

    def test_cost_history(self):
        test_cls = SimpleLinearRegressionVisualizer()
        test_cls.train(epochs=10)
        cost_hist_test = test_cls.cost_history
        cost_hist1 = self.viz1.cost_history
        self.assertIsInstance(cost_hist1, np.ndarray)
        self.assertIsInstance(cost_hist_test, np.ndarray)
        self.assertEqual(cost_hist1.shape, (0,))
        self.assertEqual(cost_hist_test.shape, (11,))
        del test_cls

    def test_weights_history(self):
        test_cls = SimpleLinearRegressionVisualizer()
        test_cls.train(epochs=10)
        weights_hist_test = test_cls.weights_history
        weights_hist1 = self.viz1.weights_history
        self.assertIsInstance(weights_hist_test, np.ndarray)
        self.assertIsInstance(weights_hist1, np.ndarray)
        self.assertEqual(weights_hist1.shape, (1, 2))
        self.assertEqual(weights_hist_test.shape, (11, 2))
        del test_cls

    def test_reset(self):
        test_cls1 = SimpleLinearRegressionVisualizer()
        test_cls2 = SimpleLinearRegressionVisualizer(randomize=True)
        test_cls3 = SimpleLinearRegressionVisualizer(no_data_points=30)
        test_cls4 = SimpleLinearRegressionVisualizer(random_state=27)
        test_cls1.train(epochs=10)
        test_cls2.train(epochs=10)
        test_cls3.train(epochs=10)
        test_cls4.train(epochs=10)
        weights_hist1, cost_hist1 = test_cls1.weights_history, test_cls1.cost_history
        weights_hist2, cost_hist2 = test_cls2.weights_history, test_cls2.cost_history
        weights_hist3, cost_hist3 = test_cls3.weights_history, test_cls3.cost_history
        weights_hist4, cost_hist4 = test_cls4.weights_history, test_cls4.cost_history
        self.assertEqual(weights_hist1.shape, (11, 2))
        self.assertEqual(weights_hist2.shape, (11, 2))
        self.assertEqual(weights_hist3.shape, (11, 2))
        self.assertEqual(weights_hist4.shape, (11, 2))
        self.assertEqual(cost_hist1.shape, (11,))
        self.assertEqual(cost_hist2.shape, (11,))
        self.assertEqual(cost_hist3.shape, (11,))
        self.assertEqual(cost_hist4.shape, (11,))
        test_cls1.reset(), test_cls2.reset(), test_cls3.reset(), test_cls4.reset()
        weights_hist1, cost_hist1 = test_cls1.weights_history, test_cls1.cost_history
        weights_hist2, cost_hist2 = test_cls2.weights_history, test_cls2.cost_history
        weights_hist3, cost_hist3 = test_cls3.weights_history, test_cls3.cost_history
        weights_hist4, cost_hist4 = test_cls4.weights_history, test_cls4.cost_history
        self.assertEqual(weights_hist1.shape, (1, 2))
        self.assertEqual(weights_hist2.shape, (1, 2))
        self.assertEqual(weights_hist3.shape, (1, 2))
        self.assertEqual(weights_hist4.shape, (1, 2))
        self.assertEqual(cost_hist1.shape, (0,))
        self.assertEqual(cost_hist2.shape, (0,))
        self.assertEqual(cost_hist3.shape, (0,))
        self.assertEqual(cost_hist4.shape, (0,))
        del test_cls1
        del test_cls2
        del test_cls3
        del test_cls4

    def test__update_weights(self):
        test_cls1 = SimpleLinearRegressionVisualizer()
        test_cls2 = SimpleLinearRegressionVisualizer(no_data_points=30)
        test_cls3 = SimpleLinearRegressionVisualizer(is_linearly_increasing=True)
        test_cls4 = SimpleLinearRegressionVisualizer(randomize=True)
        test_cls5 = SimpleLinearRegressionVisualizer(random_state=27)
        test_cls = [test_cls1, test_cls2, test_cls3, test_cls4, test_cls5]
        costs = [x.cost for x in test_cls]
        for reg in test_cls:
            [reg._update_weights() for _ in range(10)]
        new_costs = [x.cost for x in test_cls]
        self.assertEqual(np.less(new_costs, costs).all(), True)
        del test_cls1
        del test_cls2
        del test_cls3
        del test_cls4
        del test_cls5

    def test_train(self):
        test_cls1 = SimpleLinearRegressionVisualizer()
        test_cls2 = SimpleLinearRegressionVisualizer(randomize=True)
        test_cls3 = SimpleLinearRegressionVisualizer(no_data_points=30)
        test_cls4 = SimpleLinearRegressionVisualizer(random_state=27)
        test_cls1.train(epochs=10)
        test_cls2.train(epochs=10)
        test_cls3.train(epochs=10)
        test_cls4.train(epochs=10)
        weights_hist1, cost_hist1 = test_cls1.weights_history, test_cls1.cost_history
        weights_hist2, cost_hist2 = test_cls2.weights_history, test_cls2.cost_history
        weights_hist3, cost_hist3 = test_cls3.weights_history, test_cls3.cost_history
        weights_hist4, cost_hist4 = test_cls4.weights_history, test_cls4.cost_history
        self.assertEqual(weights_hist1.shape, (11, 2))
        self.assertEqual(weights_hist2.shape, (11, 2))
        self.assertEqual(weights_hist3.shape, (11, 2))
        self.assertEqual(weights_hist4.shape, (11, 2))
        self.assertEqual(cost_hist1.shape, (11,))
        self.assertEqual(cost_hist2.shape, (11,))
        self.assertEqual(cost_hist3.shape, (11,))
        self.assertEqual(cost_hist4.shape, (11,))
        del test_cls1
        del test_cls2
        del test_cls3
        del test_cls4

    def test_show_data(self):
        fig = self.viz1.show_data(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_show_initial_regression_line(self):
        fig = self.viz1.show_initial_regression_line(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_show_current_regression_line(self):
        fig = self.viz1.show_current_regression_line(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_show_regression_line_comparison(self):
        fig = self.viz1.show_regression_line_comparison(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_show_regression_line_progression(self):
        fig = self.viz1.show_regression_line_progression(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_show_cost_history(self):
        fig = self.viz1.show_cost_history(return_fig=True)
        self.assertIsInstance(fig, Figure)

    def test_visualize(self):
        anim = self.viz1.visualize()
        self.assertIsInstance(anim, FuncAnimation)
