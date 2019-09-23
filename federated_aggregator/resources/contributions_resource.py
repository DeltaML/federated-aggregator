import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

from federated_aggregator.services.federated_aggregator import FederatedAggregator

api = Namespace('contributions', description='Contributions related operations')

contributions_response = api.model(name='Contributions Request', model={
    'model_id': fields.String(required=True, description='Model id'),
    'improvement': fields.Float(required=True, description='Current improvement'),
    'contributions': fields.Raw(required=True, description='Current contributions')
})

contributions_request = api.model(name='Contributions Response', model={
    'model_id': fields.String(required=True, description='Model id'),
    'MSE': fields.Float(required=True, description='Current improvement'),
    'initial_MSE': fields.Float(required=True, description='Current improvement'),
    'partial_MSEs': fields.Raw(required=True, description='Current contributions')
})


@api.route('', endpoint='contributions_resources_ep')
class ContributionsResources(Resource):

    @api.marshal_with(contributions_response, code=200)
    @api.doc('Calculate contributions')
    @api.expect(contributions_request)
    def post(self):
        # TODO fix this
        """
            if federated_aggregator.are_valid(model_id, mse, initial_mse, partial_MSEs, public_key):
            return jsonify(federated_aggregator.calculate_contributions(model_id, mse, initial_mse, partial_MSEs))
        else:
            return jsonify({"ERROR": "Tried to falsify metrics"})  # Case when the model buyer tried to falsify
        """
        data = request.get_json()
        logging.info("Calculate Contributions {}".format(data))
        mse = data['MSE']
        partial_MSEs = data["partial_MSEs"]
        model_id = data["model_id"]
        initial_mse = data['initial_MSE']
        return FederatedAggregator().calculate_contributions(model_id, mse, initial_mse, partial_MSEs)

