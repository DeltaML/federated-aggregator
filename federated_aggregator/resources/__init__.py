import logging
from flask import jsonify, make_response
from flask_restplus import Api
from federated_aggregator.resources.register_resource import api as register_api
from federated_aggregator.resources.model_resource import api as model_api
from federated_aggregator.resources.contributions_resource import api as contributions_api
from federated_aggregator.resources.helper_resources import api as helper_api
from federated_aggregator.resources.federated_aggregator_resource import api as federated_aggregator_api
from federated_aggregator.resources.contract_resource import api as contract_api

api = Api(
    title='Federated Aggregator Api',
    version='1.0',
    description='Federated Aggregator Api API',
    doc='/doc/'
)


# Add apis to namespace
api.add_namespace(register_api)
api.add_namespace(model_api)
api.add_namespace(contributions_api)
api.add_namespace(helper_api)
api.add_namespace(federated_aggregator_api)
api.add_namespace(contract_api)


@api.errorhandler(Exception)
def default_error_handler(error):
    """
    Default error handler
    :param error:
    :return:
    """
    logging.error(error)
    return {'message': str(error)}, 500


def _handle_error(error):
    logging.error(error)
    return ErrorHandler.create_error_response(error.status_code, error.message)


class ErrorHandler:
    @staticmethod
    def create_error_response(status_code, message):
        return make_response(
            jsonify(
                {
                    "status_code": status_code,
                    "message": message
                }
            ),
            status_code
        )
