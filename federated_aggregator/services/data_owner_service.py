import logging

import numpy as np
from commons.utils.singleton import Singleton

from federated_aggregator.connectors.data_owner_connector import DataOwnerConnector
from federated_aggregator.models.data_owner_instance import DataOwnerInstance


class DataOwnerService(metaclass=Singleton):

    def __init__(self):
        self.data_owners = None
        self.encryption_service = None
        self.config = None
        self.active_encryption = None
        self.data_owner_connector = None

    def init(self, encryption_service, config):
        """
        :param encryption_service: EncriptionService
        :param config: Dict[String, Any]
        """
        self.data_owners = {}
        self.encryption_service = encryption_service
        self.config = config
        self.active_encryption = self.config["ACTIVE_ENCRYPTION"]
        self.data_owner_connector = DataOwnerConnector(self.config["DATA_OWNER_PORT"], encryption_service,
                                                       self.active_encryption)

    def register_data_owner(self, data_owner_data):
        """
        Register new data_owner
        :param data_owner_data: Dict[String, String], Keys: id, host, port
        :return: Dict[String, Int]
        """
        new_data_owner = DataOwnerInstance(data_owner_data)
        self.data_owners[new_data_owner.id] = new_data_owner
        return {'id': len(self.data_owners) - 1}

    def get_data_owners(self):
        return list(self.data_owners.values())

    def send_requirements(self, data):
        return self.data_owner_connector.send_requirements_to_data_owners(list(self.data_owners.values()), data)

    def get_linked_data_owners_to_model(self, data):
        """
        Recevies a data structure that contains the requirements over the data needed for training the model.
        Sends these requirements to the data owners. They respond each with a true if they have data that complies with
        the reqs. and false if they don't.
        This method returns a list of the data owners that have the previosly mentioned data.
        :param data:
        :return:
        """
        linked_data_owners = []
        owners_with_data = self.data_owner_connector.get_linked_data_owners(list(self.data_owners.values()), data['model_id'])
        for data_owner_link in owners_with_data:
            if (data['model_id'] == data_owner_link['model_id']) and (data_owner_link['linked']):
                data_owner_key = data_owner_link['data_owner_id']
                linked_data_owners.append(self.data_owners[data_owner_key])
        return linked_data_owners

    def send_mses(self, validators, model_data, mses, role):
        self.data_owner_connector.send_mses(validators, model_data, mses, role)

    def send_avg_gradient(self, gradient, model_data):
        """
        Sends the average gradient back to the data owners for a new gradient step.
        :param gradient: The average of the gradients received by the data owners.
        :param model_data: wrapper with data related to the current model training.
        :return: Nothing
        """
        logging.info("Send global models")
        self.data_owner_connector.send_gradient_to_data_owners(model_data.local_trainers, gradient,
                                                               model_data.model_id, model_data.public_key)

    def get_gradients(self, model_data):
        """
        :param model_data: wrapper with data related to the current model training.
        :return: the gradients calculated after a gradient descent step in the data owners, and the data owners that
        performed such calculation.
        """
        results = self.data_owner_connector.get_gradient_from_data_owners(model_data)
        gradients = np.asarray(list(map(lambda x: x['update'], results)))
        owners = list(map(lambda x: x['data_owner_id'], results))
        return gradients, owners

    def get_trained_models(self, model_data):
        """obtiene el nombre del modelo a ser entrenado"""
        logging.info("get_trained_models")
        return self.data_owner_connector.get_data_owners_model(model_data.local_trainers)

    def get_model_metrics_from_validators(self, model_data):
        logging.info("Getting model metrics from validators")
        diffs_from_validators = self.data_owner_connector.get_model_metrics_from_validators(model_data.validators,
                                                                                            model_data)
        return diffs_from_validators

    def get_partial_model_metrics_from_validators(self, partial_models, model_data):
        trainers_mses = {}
        logging.info("Getting partials mses")
        for trainer, model_weights in partial_models:
            logging.info("Calculating mse for model without trainer: {}".format(trainer))
            diffs = self.data_owner_connector.get_model_metrics_from_validators(model_data.validators, model_data,
                                                                                model_weights)

            trainers_mses[trainer] = diffs
        return trainers_mses

    def send_encrypted_prediction(self, model, encrypted_prediction):
        logging.info("send_encrypted_prediction")
        self.data_owner_connector.send_encrypted_prediction(data_owner=model.data_owner,
                                                            encrypted_prediction=encrypted_prediction)

    def send_result(self):
        self.data_owner_connector.sen