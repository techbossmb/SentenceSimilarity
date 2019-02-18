import logging

from flask import Flask, jsonify
from flask.ext.restful import reqparse
from flask_restful import Resource, Api

from test_similarity import SimilarityHelper


app = Flask(__name__)
api = Api(app)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

parser = reqparse.RequestParser()
parser.add_argument('text', type=str)
parser.add_argument('raw_text', type=str)
parser.add_argument('message', type=str)

class SimilarityApi(Resource):
    similarity_model = SimilarityHelper()

    def post(self):
        args = parser.parse_args()
        text = str(args['text'])
        response = self.similarity_model.predictSimilarity(text)
        logger.info('[Entity] Request: %s :::: Response: %s', text, response)
        return jsonify(response=response)

class ResponseApi(Resource):
    similarity_model = SimilarityHelper()

    def post(self):
        args = parser.parse_args()
        text = str(args['text'])
        response = self.similarity_model.get_answers(text)
        logger.info('[Entity] Request: %s :::: Response: %s', text, response)
        return jsonify(response=response)


api.add_resource(SimilarityApi, '/sim_api')
api.add_resource(ResponseApi, '/faq_resp')


if __name__ == '__main__':
    app.run()
