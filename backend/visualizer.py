import base64
import io

import matplotlib.pyplot as plt
from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin

# noinspection PyUnresolvedReferences
from src import SimpleLinearRegressionVisualizer

viz = Flask(__name__)
CORS(viz, support_credentials=True)


@viz.route('/home/get-topics-list', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_topics_list() -> Response:
    """
    :return: A list of topics and their corresponding subtopics.
    """

    topics = ['Supervised_Learning', 'Unsupervised_Learning', 'Neural_Networks', 'Reinforcement_Learning']
    sub_topics = [['Simple_Linear_Regression', ], [], [], []]

    return jsonify(list(zip(topics, sub_topics)))


@viz.route('/get-linear-regression-plots', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_plots() -> Response:

    output = {}
    input_data = request.get_json(force=True)

    vizualizer = SimpleLinearRegressionVisualizer(randomize=input_data['randomize'],
                                                  learning_rate=input_data['learningRate'],
                                                  no_data_points=input_data['dataPoints'],
                                                  is_linearly_increasing=input_data['linearlyIncreasing'],
                                                  no_epochs=input_data['epochs'])
    plt.switch_backend('agg')

    output['random state'] = vizualizer.random_state

    fig = vizualizer.show_data(return_fig=True)
    # output1 = io.BytesIO()
    # plt.savefig(output1, format='jpg')
    # output1.seek(0)
    # my_base64_jpgData = base64.b64encode(output1.read())
    output['data points'] = fig.to_html()

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


@viz.route('/get-linear-regression-animation', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_linear_regression_animation():

    input_data = request.get_json(force=True)
    vizualizer = SimpleLinearRegressionVisualizer(randomize=input_data['randomize'],
                                                  learning_rate=input_data['learningRate'],
                                                  no_data_points=input_data['dataPoints'],
                                                  is_linearly_increasing=input_data['linearlyIncreasing'],
                                                  no_epochs=input_data['epochs'],
                                                  random_state=input_data['randomState'])
    plt.switch_backend('agg')
    anim = vizualizer.visualize()
    html_to_render = anim.to_html5_video()

    return jsonify({'html_to_render': html_to_render})


if __name__ == '__main__':
    viz.run(host='0.0.0.0', debug=True, port=3001)
