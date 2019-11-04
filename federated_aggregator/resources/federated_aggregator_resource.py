import logging

from flask_restplus import Resource, Namespace, fields
from flask import request
from federated_aggregator.services.contract_service import ContractService

api = Namespace('federated-aggregator', description='Contract related operations')

federated_aggregator_address_req = api.model(name='Federated Aggregator Address Req', model={
    'address': fields.String(required=True, description='Federated Aggregator address')
})


federated_aggregator_address_response = api.model(name='Federated Aggregator Resp', model={
    'address': fields.String(required=True, description='Federated Aggregator address')
})


@api.route('', endpoint='federated_aggregator_account_resources_ep')
class FederatedAggregatorAccountResources(Resource):

    @api.expect(federated_aggregator_address_req)
    @api.doc('update federated aggregator account address')
    @api.marshal_with(federated_aggregator_address_response, code=200)
    def patch(self):
        logging.info("Update federated aggregator account address")
        ContractService().set_federated_aggregator_address(request.get_json()["address"])
        return ContractService().get_contract_data(), 200

    @api.doc('get federated aggregator account address')
    @api.marshal_with(federated_aggregator_address_response, code=200)
    def get(self):
        logging.info("Get federated aggregator account address")
        return ContractService().get_federated_aggregator_address(), 200
