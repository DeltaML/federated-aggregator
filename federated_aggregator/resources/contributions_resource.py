import logging

from flask import request, jsonify
from flask_restplus import Resource, Namespace, fields

from federated_aggregator.services.federated_aggregator import FederatedAggregator

api = Namespace('contributions', description='Contributions related operations')


@api.route('', endpoint='contributions_resources_ep')
class ContributionsResources(Resource):

    #@api.marshal_with(data_owner_register, code=201)
    @api.doc('Register data owner')
    #@api.expect(data_owner_register, validate=True)
    def post(self):
        data = request.get_json()
        logging.info("Data {}".format(data))
        mse = data['MSE']
        partial_MSEs = data["partial_MSEs"]
        public_key = data["public_key"]
        model_id = data["model_id"]
        initial_mse = data['initial_MSE']
        return jsonify(FederatedAggregator().calculate_contributions(model_id, mse, initial_mse, partial_MSEs))
        # TODO fix this
        """
            if federated_aggregator.are_valid(model_id, mse, initial_mse, partial_MSEs, public_key):
            return jsonify(federated_aggregator.calculate_contributions(model_id, mse, initial_mse, partial_MSEs))
        else:
            return jsonify({"ERROR": "Tried to falsify metrics"})  # Case when the model buyer tried to falsify
        """
