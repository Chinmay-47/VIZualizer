import os

from flask import Flask, jsonify, Response
from flask_cors import CORS, cross_origin

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


if __name__ == '__main__':
    viz.run(host='0.0.0.0', debug=True, port=3001)
