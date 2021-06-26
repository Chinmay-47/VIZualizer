import numpy as np
from ...utils import DataPointsGenerator, PlotUtils, timer
from typing import Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


class LinearRegressionVisualizer:
    """
    Performs and Visualizes Linear Regression.
    """

    def __init__(self, randomize: Optional[bool] = False, learning_rate: Optional[float] = 0.001,
                 no_data_points: Optional[int] = 20, is_linearly_increasing: Optional[bool] = False,
                 no_epochs: Optional[int] = 10000, random_state: Optional[int] = None):

        if not isinstance(randomize, bool):
            raise TypeError("Randomize takes only boolean values (True/False).")
        if not isinstance(learning_rate, float):
            raise TypeError("Learning rate should be a float.")
        if not isinstance(no_data_points, int):
            raise TypeError("Number of data points should be an integer.")
        if not isinstance(is_linearly_increasing, bool):
            raise TypeError("Changing of data direction takes only boolean values (True/False).")
        if not isinstance(no_epochs, int):
            raise TypeError("Number of epochs should be an integer.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("Random state should be an integer.")

        if randomize and random_state is not None:
            raise AttributeError("Cannot both randomize and set a random state.")

        self._dpgen: DataPointsGenerator = DataPointsGenerator()
        self._is_random: bool = randomize
        self.__random_state: int = 0 if random_state is None else random_state

        if self.__random_state is not None:
            self._dpgen.set_random_state(self.__random_state)

        if self._is_random:
            self.__random_state: int = np.random.randint(low=1, high=100000)
            self._dpgen.set_random_state(self.__random_state)

        self._no_data_points: int = no_data_points
        self._is_linearly_increasing: bool = is_linearly_increasing

        self._data_points: np.ndarray = self._dpgen.gen_linear2D(no_of_points=self._no_data_points,
                                                                 is_increasing=self._is_linearly_increasing)
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
        self._no_epochs = no_epochs

    @property
    def data_points(self) -> np.ndarray:
        """
        Data points used to perform and visualize Linear Regression.
        """
        return self._data_points

    @property
    def x_values(self) -> np.ndarray:
        """
        X-axis values of data points used to perform and visualize Linear Regression.
        """
        return self._x_values

    @property
    def y_values(self) -> np.ndarray:
        """
        Y-axis values of data points used to perform and visualize Linear Regression.
        """
        return self._y_values

    @property
    def theta1(self) -> float:
        """
        Slope of the regression line.
        """
        return self._theta1

    @property
    def theta0(self) -> float:
        """
        Intercept of the regression line.
        """
        return self._theta0

    @property
    def learning_rate(self) -> float:
        """
        Learning rate used in weight updating.
        """
        return self._learning_rate

    @property
    def initial_regression_line(self) -> np.ndarray:
        """
        Regression line prior to any training.
        """
        return self._initial_regression_line

    @property
    def current_regression_line(self) -> np.ndarray:
        """
        Regression line with current level of training.
        """
        return self._dpgen.gen_line_given_x(x_values=self._x_values, slope=self._theta1, intercept=self._theta0)

    @property
    def predicted_y_values(self) -> np.ndarray:
        """
        Y-axis values predicted with current weights.
        """
        return np.array(list(zip(*self._dpgen.gen_line_given_x(x_values=self._x_values, slope=self._theta1,
                                                               intercept=self._theta0)))[1])

    @property
    def cost(self) -> float:
        """
        Cost/Error of the current predicted Y-values.
        """

        self._compute_cost()
        return self._cost

    @property
    def cost_history(self) -> np.ndarray:
        """
        Cost computed before each weight update.
        """
        return np.array(self._cost_history)

    def reset(self, randomize: Optional[bool] = None, learning_rate: Optional[float] = None,
              no_data_points: Optional[int] = None, is_linearly_increasing: Optional[bool] = None,
              no_epochs: Optional[int] = None):
        """
        Resets to the initial state (before training/weight updates).
        """

        new_randomize = False if randomize is None else randomize
        new_random_state = None if new_randomize else self.__random_state
        new_learning_rate = self._learning_rate if learning_rate is None else learning_rate
        new_no_data_points = self._no_data_points if no_data_points is None else no_data_points
        new_is_lin_inc = self._is_linearly_increasing if is_linearly_increasing is None else is_linearly_increasing
        new_no_epochs = self._no_epochs if no_epochs is None else no_epochs

        self.__init__(randomize=new_randomize, learning_rate=new_learning_rate, no_data_points=new_no_data_points,
                      is_linearly_increasing=new_is_lin_inc, no_epochs=new_no_epochs, random_state=new_random_state)

    def _compute_cost(self):
        self._cost = (1/(2 * len(self._y_values))) * sum([i * i for i in (self.predicted_y_values - self._y_values)])

    def _update_weights(self):
        new_theta1 = self._theta1 - ((self._learning_rate / len(self._y_values)) *
                                     sum((self.predicted_y_values - self._y_values) * self._x_values))
        new_theta0 = self._theta0 - ((self._learning_rate / len(self._y_values)) *
                                     sum(self.predicted_y_values - self._y_values))

        self._theta1 = new_theta1
        self._theta0 = new_theta0

    @timer
    def train(self, epochs: Optional[int] = None) -> None:
        """
        Trains the model and updates weights to minimize cost/error.

        Note: Runs for 1000 epochs by default. Can be changed here or during initialization.
        """

        if epochs is None:
            epochs = self._no_epochs

        if not isinstance(epochs, int):
            raise TypeError("Number of epochs must be an integer.")

        for _ in tqdm(range(epochs)):
            self._cost_history.append(self.cost)
            self._update_weights()

    @PlotUtils.set_default_labels
    @PlotUtils.clear_prev_plots
    def show_data(self) -> None:
        """
        Shows a plot of the data points used to perform linear regression.
        """

        plt.style.use("ggplot")
        plt.scatter(list(zip(*self._data_points))[0], list(zip(*self._data_points))[1], marker='*', c='red')

    @PlotUtils.set_default_labels
    @PlotUtils.clear_prev_plots
    def show_initial_regression_line(self, include_data: Optional[bool] = True) -> None:
        """
        Shows a plot of the initial regression line with or without data.
        """

        plt.style.use("ggplot")
        if include_data:
            plt.scatter(list(zip(*self._data_points))[0], list(zip(*self._data_points))[1], marker='*', c='red')
        plt.plot(list(zip(*self._initial_regression_line))[0],
                 list(zip(*self._initial_regression_line))[1], c='blue')

    @PlotUtils.set_default_labels
    @PlotUtils.clear_prev_plots
    def show_current_regression_line(self, include_data: Optional[bool] = True) -> None:
        """
        Shows a plot of the current regression line with or without data.
        """

        plt.style.use("ggplot")
        if include_data:
            plt.scatter(list(zip(*self._data_points))[0], list(zip(*self._data_points))[1], marker='*', c='red')
        plt.plot(list(zip(*self.current_regression_line))[0],
                 list(zip(*self.current_regression_line))[1], c='blue')

    @PlotUtils.set_default_labels
    @PlotUtils.clear_prev_plots
    def show_regression_line_comparison(self, include_data: Optional[bool] = True) -> None:
        """
        Shows a plot of the current regression line with or without data.
        """

        plt.style.use("ggplot")
        if include_data:
            plt.scatter(list(zip(*self._data_points))[0], list(zip(*self._data_points))[1], marker='*', c='red')
        plt.plot(list(zip(*self._initial_regression_line))[0],
                 list(zip(*self._initial_regression_line))[1], c='blue', label="Intial")
        plt.plot(list(zip(*self.current_regression_line))[0],
                 list(zip(*self.current_regression_line))[1], c='green', label="Trained")
        plt.title("Initial Regression Line vs Trained Regression Line")
        plt.legend()

    @PlotUtils.clear_prev_plots
    def show_cost_history(self) -> None:
        """
        Shows a plot of the cost through the history of training.
        """

        plt.style.use("ggplot")
        plt.plot(list(range(len(self._cost_history))), self._cost_history, c='blue')
        plt.title("Cost History")
        plt.xlabel("Epochs")
        plt.ylabel("Cost")

    @PlotUtils.clear_prev_plots
    def visualize(self, show_data: Optional[bool] = True, show_initial: Optional[bool] = True,
                  save: Optional[bool] = False, save_file: Optional[str] = None) -> Union[FuncAnimation, None]:
        """
        Visualizes the process of Linear regression.

        :param show_data: Can choose to not display data points.
        :param show_initial: Can choose to not display initial regression line.
        :param save: Can save the animation as a video in the current working directory.
        :param save_file: Name of the file to save.
        :return: Animation Object.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        if show_data:
            ax.scatter(list(zip(*self._data_points))[0], list(zip(*self._data_points))[1], marker='*', c='red')
        if show_initial:
            ax.plot(list(zip(*self.initial_regression_line))[0], list(zip(*self.initial_regression_line))[1],
                    c='blue', label="Intial")
        line, = ax.plot(list(zip(*self.current_regression_line))[0],
                        list(zip(*self.current_regression_line))[1],
                        scaley=False, scalex=False, c='green', label="Trained")
        ax.legend()

        def _animate(i):
            if i >= ((self._no_epochs//100) * 0.85):
                [self._update_weights() for _ in range(1000)]
            elif i >= ((self._no_epochs//100) * 0.85):
                [self._update_weights() for _ in range(500)]
            elif i >= ((self._no_epochs//100) * 0.75):
                [self._update_weights() for _ in range(300)]
            elif i >= ((self._no_epochs//100) * 0.6):
                [self._update_weights() for _ in range(200)]
            elif i >= ((self._no_epochs//100) * 0.5):
                [self._update_weights() for _ in range(150)]
            elif i >= ((self._no_epochs//100) * 0.4):
                [self._update_weights() for _ in range(100)]
            elif i >= ((self._no_epochs//100) * 0.2):
                [self._update_weights() for _ in range(20)]
            if i >= ((self._no_epochs//100) * 0.1):
                [self._update_weights() for _ in range(5)]
            else:
                self._update_weights()
            line.set_ydata(list(zip(*self.current_regression_line))[1])
            return line,

        @PlotUtils.set_default_labels
        def _init_func():
            self.reset()
            line.set_ydata(list(zip(*self.current_regression_line))[1])
            return line,

        animation = FuncAnimation(fig, _animate, interval=1, blit=True, save_count=50, init_func=_init_func,
                                  repeat=True, frames=(self._no_epochs//100), repeat_delay=1000)
        plt.show()

        if save:
            if save_file is None:
                animation.save("Linear_Regression_Visualization.gif", writer="Pillow", fps=30, bitrate=36000)
            PlotUtils.clear_plots()
            return

        return animation
