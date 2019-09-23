import logging
import os
from logging.config import dictConfig

from commons.encryption.encryption_service import EncryptionService
from flask import Flask

from federated_aggregator.config.logging_config import PROD_LOGGING_CONFIG, DEV_LOGGING_CONFIG
from federated_aggregator.resources import api
from federated_aggregator.services.data_owner_service import DataOwnerService
from federated_aggregator.services.federated_aggregator import FederatedAggregator


def create_app():
    # create and configure the app
    flask_app = Flask(__name__)
    if 'ENV_PROD' in os.environ and os.environ['ENV_PROD']:
        flask_app.config.from_pyfile("config/prod/app_config.py")
        dictConfig(PROD_LOGGING_CONFIG)
        logging.info("Using prod config")
    else:
        dictConfig(DEV_LOGGING_CONFIG)
        flask_app.config.from_pyfile("config/dev/app_config.py")
        logging.info("Using dev config")
    # ensure the instance folder exists
    try:
        os.makedirs(flask_app.instance_path)
    except OSError:
        pass
    return flask_app


app = create_app()
api.init_app(app)
encryption_service = EncryptionService(is_active=app.config["ACTIVE_ENCRYPTION"])
data_owner_service = DataOwnerService()
federated_aggregator = FederatedAggregator()
data_owner_service.init(encryption_service, app.config)
federated_aggregator.init(encryption_service=encryption_service,
                          data_owner_service=data_owner_service,
                          config=app.config)

logging.info("federated_aggregator running")
