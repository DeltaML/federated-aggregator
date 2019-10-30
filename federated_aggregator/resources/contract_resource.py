import logging

from flask_restplus import Resource, Namespace, fields
from flask import request
from federated_aggregator.services.contract_service import ContractService

api = Namespace('contract', description='Contract related operations')

contract_address_req = api.model(name='Contract Address Req', model={
    'address': fields.String(required=True, description='New contract address')
})


contract_address_response = api.model(name='Contract Address Resp', model={
    'address': fields.String(required=True, description='The contract address')
})


@api.route('', endpoint='contract_resources_ep')
class ContractResources(Resource):

    @api.expect(contract_address_req)
    @api.doc('update contract address')
    @api.marshal_with(contract_address_response, code=200)
    def patch(self):
        logging.info("Update contract address")
        ContractService().set_contract_address(request.get_json()["address"])
        return ContractService().get_contract_data(), 200

    @api.doc('get contract address')
    @api.marshal_with(contract_address_response, code=200)
    def get(self):
        logging.info("Get contract address")
        return ContractService().get_contract_data(), 200

