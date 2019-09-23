import logging
from flask import request
from flask_restplus import Resource, Namespace, fields

from federated_aggregator.services.data_owner_service import DataOwnerService

api = Namespace('dataowner', description='Data Owner related operations')


data_owner_register = api.model(name='Register', model={
    'id': fields.String(required=True, description='Data owner subscription id')
})


data_owner = api.model(name='Data Owner', model={
    'id': fields.String(required=True, description='Data owner registered id'),
    'host': fields.String(required=True, description='Data owner registered host'),
    'port': fields.String(required=True, description='Data owner registered port')
})


@api.route('', endpoint='dataowner_resources_ep')
class DataOwnerResources(Resource):

    @api.marshal_with(data_owner_register, code=201)
    @api.doc('Register data owner')
    @api.expect(data_owner_register, validate=True)
    def post(self):
        logging.info("Register data owner")
        data = request.get_json()
        data["host"], data["port"] = request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']
        return DataOwnerService().register_data_owner(data)

    @api.marshal_list_with(data_owner)
    def get(self):
        logging.info("Get data owners")
        return DataOwnerService().get_data_owners()
