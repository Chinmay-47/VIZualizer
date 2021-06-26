import os
import io
from flask import Flask, jsonify, Response
from flask_cors import CORS, cross_origin
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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


@viz.route('/home/linear-regression', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_linear_Regression_plots() -> Response:

    vizualizer = LinearRegressionVisualizer()
    fig1 = vizualizer.show_data(return_fig=True)
    # fig2 = vizualizer.show_initial_regression_line(return_fig=True)
    # vizualizer.train()
    # fig3 = vizualizer.show_regression_line_comparison(return_fig=True)
    output1 = io.BytesIO()
    # output2 = io.BytesIO()
    # output3 = io.BytesIO()
    FigureCanvas(fig1).print_png(output1)
    # FigureCanvas(fig2).print_png(output2)
    # FigureCanvas(fig3).print_png(output3)
    return Response(output1.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    viz.run(host='0.0.0.0', debug=True, port=3001)
