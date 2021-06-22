import numpy as np
from ...utils import DataPointsGenerator
from typing import Optional


class LinearRegressionVisualizer:
    """Performs and Visualizes Linear Regression"""

    def __init__(self, randomize: Optional[bool] = False, learning_rate: Optional[float] = 0.01,
                 no_data_points: Optional[int] = 20, linearly_increasing: Optional[bool] = True,
                 no_epochs: Optional[int] = 1000):

        if not isinstance(randomize, bool):
            raise TypeError("Randomize takes only boolean values (True/False).")
        if not isinstance(learning_rate, float):
            raise TypeError("Learning rate should be a float.")
        if not isinstance(no_data_points, int):
            raise TypeError("Number of data points should be an integer.")
        if not isinstance(linearly_increasing, bool):
            raise TypeError("Changing of data direction takes only boolean values (True/False).")

        self._dpgen: DataPointsGenerator = DataPointsGenerator()

        if randomize:
            self._dpgen.set_random_state(np.random.randint(low=1, high=100))

        self._no_data_points: int = no_data_points

        self._data_points: np.ndarray = self._dpgen.gen_linear2D(no_of_points=self._no_data_points,
                                                                 is_increasing=linearly_increasing)
        self._x_values: np.ndarray = np.array(list(zip(*self._data_points))[0])
        self._y_values: np.ndarray = np.array(list(zip(*self._data_points))[1])
        self._theta1: float = self._dpgen.gen_float()
        self._theta0: float = self._dpgen.gen_float()
        self._learning_rate: float = learning_rate
        self._cost: float = float()
        self._cost_history: list = list()
        self._initial_regression_line: np.ndarray = self._dpgen.gen_line_given_x(x_values=self._x_values,
                                                                                 slope=self._theta1,
                                                                                 intercept=self._theta0)

    @property
    def data_points(self):
        return self._data_points

    @property
    def x_values(self):
        return self._x_values

    @property
    def y_values(self):
        return self._y_values

    @property
    def theta1(self):
        return self._theta1

    @property
    def theta0(self):
        return self._theta0

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def initial_regression_line(self) -> np.ndarray:
        return self._initial_regression_line

    @property
    def current_regression_line(self) -> np.ndarray:
        return self._dpgen.gen_line_given_x(x_values=self._x_values, slope=self._theta1, intercept=self._theta0)

    @property
    def predicted_y_values(self) -> np.ndarray:
        return np.array(list(zip(*self._dpgen.gen_line_given_x(x_values=self._x_values, slope=self._theta1,
                                                               intercept=self._theta0)))[1])

    @property
    def cost(self) -> float:
        self._compute_cost()
        return self._cost

    @property
    def cost_history(self) -> np.ndarray:
        return np.array(self._cost_history)

    def _compute_cost(self):
        self._cost = (1/(2 * len(self._y_values))) * sum([i * i for i in (self._y_values - self.predicted_y_values)])

    def _update_theta1(self, new_weight: float) -> None:
        self._theta1 = new_weight

    def _update_theta0(self, new_weight: float) -> None:
        self._theta0 = new_weight

    def _update_weights(self):
        new_theta1 = self._theta1 - ((self._learning_rate / len(self._y_values)) *
                                     sum((self.predicted_y_values - self._y_values) * self._x_values))
        new_theta0 = self._theta0 - ((self._learning_rate / len(self._y_values)) *
                                     sum(self.predicted_y_values - self._y_values))

        self._cost_history.append(self.cost)
        self._update_theta1(new_theta1)
        self._update_theta0(new_theta0)
