from typing import Optional, Union

import numpy as np


class DataPointsGenerator:
    """DataPointsGenerator: Generates Data points of different dimensions and distributions."""

    __random_state: int = 42

    @classmethod
    def set_random_state(cls, random_state: int) -> None:
        """
        Sets the random state of the DataPointsGenerator.
        This random state value is used as the seed value in the data point generator functions.

        :param random_state: The value of the the random state.
        """
        if not isinstance(random_state, int):
            raise TypeError("Random State needs to be an Integer.")

        cls.__random_state = random_state

    def __setattr__(self, key, value):
        if key == '__random_state':
            raise AttributeError("The value of random state has already been set. "
                                 "Use the 'set_random_state' method to re-set it to a new value.")
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key == '__random_state':
            raise AttributeError('The random state attribute can not be deleted')
        else:
            del self.__dict__[key]

    def __str__(self):
        return self.__doc__

    def __repr__(self):
        return self.__class__.__name__ + "()"

    @property
    def random_state(self):
        """
        The value of the random state.
        """
        return self.__random_state

    def __set_seed(self) -> None:
        """
        Sets the seed value to be equal the random state.
        This seed value is used in the data point generator functions.
        """
        np.random.seed(seed=self.__random_state)

    def gen_float(self, randomize: Optional[bool] = False) -> float:
        """
        Generates a float between 0 and 1.

        :param randomize: Will generate a new float if set to True.
        :return: A float between 0 and 1 excluding 1.
        """

        if not randomize:
            self.__set_seed()

        return np.random.random()

    def gen_normal_1D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 1-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 1-D points or a single 1-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(no_of_points)

    def gen_normal_2D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 2-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 2-D points or a single 2-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(size=(no_of_points, 2))

    def gen_normal_3D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 3-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 3-D points or a single 3-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(size=(no_of_points, 3))

    def gen_linear1D(self, no_of_points: Optional[int] = 1, is_increasing: Optional[bool] = True,
                     randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 1-Dimensional Data in a linearly ascending or descending order.

        :param no_of_points: Shape of the array of points to be returned.
        :param is_increasing: True by default. Returns decreasing, if set to False.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 1 Dimensional linearly ascending or descending points.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not isinstance(is_increasing, bool):
            raise TypeError("Input for is_increasing needs to be True or False.")

        if not randomize:
            self.__set_seed()

        if not is_increasing:
            return np.array([i + np.random.standard_normal() for i in reversed(range(no_of_points))])

        return np.array([i + np.random.standard_normal() for i in range(no_of_points)])

    def gen_linear2D(self, no_of_points: Optional[int] = 1, is_increasing: Optional[bool] = True,
                     randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 2-Dimensional Data in a linearly ascending or descending order.

        :param no_of_points: Shape of the array of points to be returned.
        :param is_increasing: True by default. Returns decreasing, if set to False.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 2 Dimensional linearly ascending or descending points.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not isinstance(is_increasing, bool):
            raise TypeError("Input for is_increasing needs to be True or False.")

        if not randomize:
            self.__set_seed()

        if not is_increasing:
            return np.array([np.array([i + np.random.standard_normal(), j + np.random.standard_normal()])
                             for i, j in list(zip(range(no_of_points), reversed(range(no_of_points))))])

        return np.array([np.array([i + np.random.standard_normal(), i + np.random.standard_normal()])
                         for i in range(no_of_points)])

    @staticmethod
    def gen_line(slope: Union[float, int], intercept: Union[float, int], no_points: Optional[int] = 10) -> np.ndarray:
        """
        Generates a line for a given slope and intercept as y = mx + c

        :param slope: Slope of the line to construct.
        :param intercept: Intercept of the line to construct.
        :param no_points: Number of points to use to construct the line.
        :return: A numpy array of points (x, y) on the line.
        """

        if not isinstance(slope, (float, int)):
            raise TypeError("Slope should be either a float or an integer.")
        if not isinstance(intercept, (float, int)):
            raise TypeError("Intercept should be either a float or an integer.")

        x_vals = np.linspace(1, no_points, no_points)
        y_vals = (slope * x_vals) + intercept

        return np.array(list(zip(x_vals, y_vals)))
