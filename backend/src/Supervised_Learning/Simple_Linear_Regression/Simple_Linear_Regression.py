from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# noinspection PyUnresolvedReferences
from src.utils import (DataPointsGenerator, timer, clear_prev_plots, set_default_labels, clear_plots,
                       return_or_save_figure)
from tqdm import tqdm


class SimpleLinearRegressionVisualizer:
    """
    Performs and Visualizes Simple Linear Regression.
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
        self._x_values: np.ndarray = self._data_points[:, 0]
        self._y_values: np.ndarray = self._data_points[:, 1]
        self._theta1: float = self._dpgen.gen_float()
        self._theta0: float = self._dpgen.gen_float()
        self._learning_rate: float = learning_rate
        self._cost: float = float()
        self._cost_history: list = list()
        self._weights_history: list = [(self._theta1, self._theta0)]
        self._initial_regression_line: np.ndarray = self._dpgen.gen_line_given_x(x_values=self._x_values,
                                                                                 slope=self._theta1,
                                                                                 intercept=self._theta0)
        self._no_epochs = no_epochs

    @property
    def random_state(self) -> int:
        """
        Random state used to generate all initial values for all parameters.
        """
        return self.__random_state

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
        return self.current_regression_line[:, 1]

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

    @property
    def weights_history(self) -> np.ndarray:
        """
        Weight before each weight update.
        """
        return np.array(self._weights_history)

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

        _err1, _err0 = [], []
        _errs1 = ((self.predicted_y_values - self._y_values) * self._x_values)
        _errs0 = (self.predicted_y_values - self._y_values)

        # We cap the max and min values to avoid overflow during any operations.
        for i, j in zip(_errs1, _errs0):
            if -1.7976931348623157e+150 < i < 1.7976931348623157e+150:
                _err1.append(i)
            elif i < -1.7976931348623157e+150:
                _err1.append(-1.7976931348623157e+150)
            elif i > 1.7976931348623157e+150:
                _err1.append(1.7976931348623157e+150)
            if -1.7976931348623157e+150 < j < 1.7976931348623157e+150:
                _err0.append(j)
            elif j < -1.7976931348623157e+150:
                _err0.append(-1.7976931348623157e+150)
            elif j > 1.7976931348623157e+100:
                _err0.append(1.7976931348623157e+150)

        upd1_ = ((self._learning_rate / len(self._y_values)) * sum(_err1))
        upd0_ = ((self._learning_rate / len(self._y_values)) * sum(_err0))

        # We do not update the weights to prevent gradients from exploding out of bounds and cause overflow.
        if upd1_ < -1.7976931348623157e+300 or upd1_ > 1.7976931348623157e+300:
            upd1_ = 0.0

        if upd0_ < -1.7976931348623157e+300 or upd0_ > 1.7976931348623157e+300:
            upd0_ = 0.0

        new_theta1 = self._theta1 - upd1_
        new_theta0 = self._theta0 - upd0_

        self._theta1 = new_theta1
        self._theta0 = new_theta0

    @timer
    def train(self, epochs: Optional[int] = None) -> None:
        """
        Trains the model and updates weights to minimize cost/error.

        Note: Runs for 10000 epochs by default. Can be changed here or during initialization.
        """

        if epochs is None:
            epochs = self._no_epochs

        if not isinstance(epochs, int):
            raise TypeError("Number of epochs must be an integer.")

        # Initial cost
        if not self._cost_history:
            self._cost_history.append(self.cost)

        for _ in tqdm(range(epochs)):
            self._update_weights()
            self._weights_history.append((self._theta1, self._theta0))
            self._cost_history.append(self.cost)

    def show_data(self, **kwargs) -> Optional[Figure]:
        """
        Shows a plot of the data points used to perform linear regression.

        Pass save=True as a keyword argument to save figure.

        Pass return_fig=True as a keyword argument to return the figure.
        """

        fig = go.Figure(data=[go.Scatter(x=self._x_values, y=self._y_values, mode='markers',
                                         marker=dict(size=8, color='red', opacity=0.8))])
        fig.update_layout(
            title="Linear Regression Data",
            xaxis_title="X Values",
            yaxis_title="Y Values",
            title_x=0.5
        )

        if 'save' in kwargs and kwargs['save']:
            fig.write_image(self.__class__.__name__ + '_' + self.show_data.__name__ + '.jpeg')

        if 'return_fig' in kwargs and kwargs['return_fig']:
            return fig

        fig.show()

    @return_or_save_figure
    @set_default_labels
    @clear_prev_plots
    def show_initial_regression_line(self, include_data: Optional[bool] = True, **kwargs) -> Optional[Figure]:
        """
        Shows a plot of the initial regression line with or without data.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        if include_data:
            ax.scatter(self._x_values, self._y_values, marker='*', c='red')
        ax.plot(self._initial_regression_line[:, 0], self._initial_regression_line[:, 1], c='blue')
        plt.title("Initial Regression Line")

        if kwargs['return_fig']:
            return fig

    @return_or_save_figure
    @set_default_labels
    @clear_prev_plots
    def show_current_regression_line(self, include_data: Optional[bool] = True, **kwargs) -> Optional[Figure]:
        """
        Shows a plot of the current regression line with or without data.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        if include_data:
            ax.scatter(self._x_values, self._y_values, marker='*', c='red')
        ax.plot(self.current_regression_line[:, 0], self.current_regression_line[:, 1], c='blue')
        plt.title("Current Regression Line")

        if kwargs['return_fig']:
            return fig

    @return_or_save_figure
    @set_default_labels
    @clear_prev_plots
    def show_regression_line_comparison(self, include_data: Optional[bool] = True, **kwargs) -> Optional[Figure]:
        """
        Shows a plot of the current regression line with or without data.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        if include_data:
            ax.scatter(self._x_values, self._y_values, marker='*', c='red')
        ax.plot(self._initial_regression_line[:, 0], self._initial_regression_line[:, 1], c='blue', label="Intial")
        ax.plot(self.current_regression_line[:, 0], self.current_regression_line[:, 1], c='green', label="Trained")
        plt.title("Initial Regression Line vs Trained Regression Line")
        ax.legend()

        if kwargs['return_fig']:
            return fig

    @return_or_save_figure
    @clear_prev_plots
    def show_regression_line_progression(self, include_data: Optional[bool] = True, **kwargs) -> Optional[Figure]:
        """
        Shows a collage of the regression line progression through training.
        """
        plt.style.use("ggplot")
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all')
        _all_ax = [ax1, ax2, ax3, ax4, ax5, ax6]

        if include_data:
            [axis.scatter(self._x_values, self._y_values, marker='*', c='red') for axis in _all_ax]

        ax1.plot(self._initial_regression_line[:, 0], self._initial_regression_line[:, 1], c='green')
        ax1.set_title("No Training",  fontsize=8)

        _weights = []
        for i in range(1, 6):
            _slope = self.weights_history[((self.weights_history.shape[0] - 1)//5) * i][0]
            _intercept = self.weights_history[((self.weights_history.shape[0] - 1)//5) * i][1]
            _curr_line = self._dpgen.gen_line(slope=_slope, intercept=_intercept, no_points=self._no_data_points)
            _all_ax[i].plot(_curr_line[:, 0], _curr_line[:, 1], c='green')
            _all_ax[i].set_title("{}% Trained".format(i * 20), fontsize=8)

        fig.suptitle("Regression Line Progression", fontsize='x-large')

        if kwargs['return_fig']:
            return fig

    @return_or_save_figure
    @clear_prev_plots
    def show_cost_history(self, **kwargs) -> Optional[Figure]:
        """
        Shows a plot of the cost through the history of training.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self._cost_history))), self._cost_history, c='blue')
        plt.title("Cost History")
        plt.xlabel("Epochs")
        plt.ylabel("Cost")

        if kwargs['return_fig']:
            return fig

    @clear_prev_plots
    def visualize(self, show_data: Optional[bool] = True, show_initial: Optional[bool] = True,
                  save: Optional[bool] = False) -> Optional[FuncAnimation]:
        """
        Visualizes the process of Linear regression.

        :param show_data: Can choose to not display data points.
        :param show_initial: Can choose to not display initial regression line.
        :param save: Can save the animation as a video in the current working directory.
        :return: Animation Object.
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        if show_data:
            ax.scatter(self._x_values, self._y_values, marker='*', c='red')
        line, = ax.plot(self.current_regression_line[:, 0], self.current_regression_line[:, 1],
                        scaley=False, scalex=False, c='green', label="Trained")
        if show_initial:
            ax.plot(self._initial_regression_line[:, 0], self._initial_regression_line[:, 1], c='blue', label="Intial")
        ax.legend()

        def _animate(i):
            if i < 10:
                self._update_weights()
            elif i < 30:
                [self._update_weights() for _ in range(int(0.0025 * self._no_epochs))]
            elif i < 40:
                [self._update_weights() for _ in range(int(0.005 * self._no_epochs))]
            elif i < 60:
                [self._update_weights() for _ in range(int(0.01 * self._no_epochs))]
            else:
                [self._update_weights() for _ in range(int(0.02 * self._no_epochs))]

            line.set_ydata(self.current_regression_line[:, 1])
            return line,

        @set_default_labels
        def _init_func():
            self.reset()
            line.set_ydata(self.current_regression_line[:, 1])
            return line,

        animation = FuncAnimation(fig, _animate, interval=40, blit=True, save_count=50, init_func=_init_func,
                                  repeat=True, frames=100, repeat_delay=8000)
        plt.show()

        if save:
            Writer = FFMpegWriter(fps=30, codec='libx264', bitrate=-1)
            animation.save("Simple_Linear_Regression_Visualization.mp4", writer=Writer)
            clear_plots()
            return

        return animation


def main():
    viz = SimpleLinearRegressionVisualizer()
    viz.visualize()


def _profiler():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        viz = SimpleLinearRegressionVisualizer()
        viz.train()
        viz.show_cost_history()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__ == '__main__':
    # _profiler()
    main()
