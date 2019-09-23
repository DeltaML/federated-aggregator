import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

from federated_aggregator.services.federated_aggregator import FederatedAggregator

api = Namespace('model', description='Model related operations')

model_schema = api.model(name='Model', model={
    'buyer_id': fields.String(required=True, description='Model buyer id'),
    'model_id': fields.String(required=True, description='Model id'),
    'buyer_host': fields.String(required=True, description='Model buyer host'),
    'model_type': fields.String(required=True, description='Model type'),
    'model_status': fields.String(required=True, description='Model status'),
    'initial_mse': fields.Float(required=True, description='Initial mse'),
    'mse': fields.Float(required=True, description='Current mse')
})


@api.route('', endpoint='model_resources_ep')
class ModelResources(Resource):

    @api.doc('Train model')
    def post(self):
        data = request.get_json()
        logging.info("Initializing async model training according to request {}".format(data))
        logging.info("host {} port {}".format(request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']))
        FederatedAggregator().process(request.environ['REMOTE_ADDR'], data)
        return

    @api.doc('Get trained model')
    @api.marshal_list_with(model_schema)
    def get(self):
        logging.info("Get models")
        return FederatedAggregator().get_models()
