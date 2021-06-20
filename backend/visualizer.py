from flask import Flask, jsonify
import os

viz = Flask(__name__)


@viz.route('/')
def index():
    return "Hello World!"


@viz.route('/home/supervised/list', methods=['GET'])
def supervised_list():
    sup_list = os.listdir('Supervised_Learning/')
    return jsonify([x.replace('_', ' ') for x in sup_list if x!= '__init__.py'])


@viz.route('/home/unsupervised/list', methods=['GET'])
def unsupervised_list():
    sup_list = os.listdir('Unsupervised_Learning/')
    return jsonify([x.replace('_', ' ') for x in sup_list if x!= '__init__.py'])


@viz.route('/home/reinforcement/list', methods=['GET'])
def reinforcement_list():
    sup_list = os.listdir('Reinforcement_Learning/')
    return jsonify([x.replace('_', ' ') for x in sup_list if x!= '__init__.py'])


@viz.route('/home/neural/list', methods=['GET'])
def neural_list():
    sup_list = os.listdir('Neural_Networks/')
    return jsonify([x.replace('_', ' ') for x in sup_list if x!= '__init__.py'])


if __name__ == '__main__':
    viz.run(host='0.0.0.0', debug=True, port=3001)
