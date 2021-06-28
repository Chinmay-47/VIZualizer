import base64
import io
import os

import matplotlib.pyplot as plt
from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin

from .Supervised_Learning.Linear_Regression.Linear_Regression import LinearRegressionVisualizer

viz = Flask(__name__)
CORS(viz, support_credentials=True)


@viz.route('/home/get-topics-list', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_topics_list() -> Response:
    """
    :return: A list of topics and their corresponding subtopics.
    """

    topics = [topic.name for topic in os.scandir(os.getcwd()) if topic.is_dir() and topic.name != '__pycache__']
    sub_topics = [[sub_topic.name for sub_topic in os.scandir(os.getcwd() + '/' + topic) if sub_topic.is_dir()
                   and sub_topic.name != '__pycache__'] for topic in topics]

    return jsonify(list(zip(topics, sub_topics)))


@viz.route('/get-linear-regression-plots', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_plots() -> Response:

    output = {}
    input_data = request.get_json(force=True)

    vizualizer = LinearRegressionVisualizer(randomize=input_data['randomize'],
                                            learning_rate=input_data['learningRate'],
                                            no_data_points=input_data['dataPoints'],
                                            is_linearly_increasing=input_data['linearlyIncreasing'],
                                            no_epochs=input_data['epochs'])
    plt.switch_backend('agg')

    _ = vizualizer.show_data(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())
    output['data points'] = str(my_base64_jpgData)

    _ = vizualizer.show_initial_regression_line(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())
    output['initial regression line'] = str(my_base64_jpgData)

    vizualizer.train()

    _ = vizualizer.show_regression_line_comparison(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())
    output['regression line comparison'] = str(my_base64_jpgData)

    _ = vizualizer.show_regression_line_progression(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())
    output['regression line progression'] = str(my_base64_jpgData)

    _ = vizualizer.show_cost_history(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())
    output['cost history'] = str(my_base64_jpgData)

    return jsonify(output)


@viz.route('/get-linear-regression-data', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_data() -> Response:

    input_data = request.get_json(force=True)

    vizualizer = LinearRegressionVisualizer(randomize=input_data['randomize'],
                                            learning_rate=input_data['learningRate'],
                                            no_data_points=input_data['dataPoints'],
                                            is_linearly_increasing=input_data['linearlyIncreasing'],
                                            no_epochs=input_data['epochs'])
    plt.switch_backend('agg')
    _ = vizualizer.show_data(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())

    return jsonify({'base64String': str(my_base64_jpgData)})


@viz.route('/get-linear-regression-initial-line', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_initial_line() -> Response:

    input_data = request.get_json(force=True)

    vizualizer = LinearRegressionVisualizer(randomize=input_data['randomize'],
                                            learning_rate=input_data['learningRate'],
                                            no_data_points=input_data['dataPoints'],
                                            is_linearly_increasing=input_data['linearlyIncreasing'],
                                            no_epochs=input_data['epochs'])
    plt.switch_backend('agg')

    _ = vizualizer.show_initial_regression_line(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())

    return jsonify({'base64String': str(my_base64_jpgData)})


@viz.route('/get-linear-regression-comparison', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_final_line() -> Response:

    input_data = request.get_json(force=True)

    vizualizer = LinearRegressionVisualizer(randomize=input_data['randomize'],
                                            learning_rate=input_data['learningRate'],
                                            no_data_points=input_data['dataPoints'],
                                            is_linearly_increasing=input_data['linearlyIncreasing'],
                                            no_epochs=input_data['epochs'])
    plt.switch_backend('agg')

    vizualizer.train()

    plt.switch_backend('agg')
    _ = vizualizer.show_regression_line_comparison(return_fig=True)
    output1 = io.BytesIO()
    plt.savefig(output1, format='jpg')
    output1.seek(0)
    my_base64_jpgData = base64.b64encode(output1.read())

    return jsonify({'base64String': str(my_base64_jpgData)})


if __name__ == '__main__':
    viz.run(host='0.0.0.0', debug=True, port=3001)
