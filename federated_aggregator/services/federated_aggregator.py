import logging
import random
from copy import deepcopy
from functools import reduce
from threading import Thread

import numpy as np
from commons.model.model_service import ModelFactory
from commons.operations_utils.functions import serialize, deserialize
from commons.operations_utils.functions import sum_collection
from commons.utils.singleton import Singleton
from commons.web3.delta_contracts import FederatedAggregatorContract

from federated_aggregator.connectors.model_buyer_connector import ModelBuyerConnector
from federated_aggregator.exceptions.exceptions import EmptyValidatorsException
from federated_aggregator.models.global_model import GlobalModel


class FederatedAggregator(metaclass=Singleton):
    def __init__(self):
        self.encryption_service = None
        self.data_owner_service = None
        self.smart_contract = None
        self.config = None
        self.eth_address = None
        self.active_encryption = None
        self.model_buyer_connector = None
        self.global_models = {}
        self.split_coefficient = None
        self.n_iter = None
        self.n_iter_partial_res = None

    def init(self, encryption_service, data_owner_service, w3_service, config):
        """
        :param w3_service:
        :param data_owner_service:
        :param encryption_service: EncriptionService
        :param config: Dict[String, Any]
        """
        self.config = config
        self.eth_address = self.config["FEDERATED_AGGREGATOR_ADDRESS"]
        self.encryption_service = encryption_service
        self.data_owner_service = data_owner_service
        self.smart_contract = FederatedAggregatorContract(
            contract=w3_service.build_contract(abi_path=self.config["ABI_PATH"],
                                               address=self.config["CONTRACT_ADDRESS"]))
        self.active_encryption = self.config["ACTIVE_ENCRYPTION"]
        self.model_buyer_connector = ModelBuyerConnector(self.config["MODEL_BUYER_HOST"],
                                                         self.config["MODEL_BUYER_PORT"])
        self.global_models = {}
        self.split_coefficient = self.config["SPLIT_COEFFICIENT"]
        self.n_iter = self.config["MAX_ITERATIONS"]
        self.n_iter_partial_res = self.config["ITERS_UNTIL_PARTIAL_RESULT"]

    def get_models(self):
        return list(self.global_models.values())

    def process(self, remote_address, data):
        Thread(target=self.async_server_processing,
               args=self._build_async_processing_data(data, remote_address)).start()

    @staticmethod
    def async_server_processing(func, *args):
        logging.info("process_in_background...")
        func(*args)
        logging.info("finish process_in_background...")

    def _build_async_processing_data(self, data, remote_address):
        data["remote_address"] = remote_address
        return self.federated_learning_wrapper, data

    def federated_learning(self, data):
        logging.info("Init federated_learning")
        model_id = data['model_id']
        try:
            linked_data_owners = self.data_owner_service.link_data_owners_to_model(data)
            self.validate_linked_data_owners(linked_data_owners, model_id)
            local_trainers, validators = self.split_data_owners(linked_data_owners)
            self.encryption_service.set_public_key(data["public_key"])
            model = self.initialize_global_model(data)
            self.global_models[model_id] = GlobalModel(model_id=model_id,
                                                       buyer_id=data["model_buyer_id"],
                                                       buyer_host=data["remote_address"],
                                                       model_type=data['model_type'],
                                                       model_status=data["status"],
                                                       local_trainers=local_trainers,
                                                       validators=validators,
                                                       model=model,
                                                       initial_mse=None,
                                                       mse=None,
                                                       public_key=data["public_key"],
                                                       partial_MSEs=None,
                                                       step=data["step"])
            # Init smart contract
            self.init_contract(model_id, data["model_buyer_address"], local_trainers, validators)
            logging.info('Running distributed gradient aggregation for {:d} iterations'.format(self.n_iter))
            model_data = self.global_models[model_id]
            diffs = self.data_owner_service.get_model_metrics_from_validators(model_data)
            model_update = self.send_partial_result_to_model_buyer(model_data, diffs, {}, True)
            mses = [np.mean(np.asarray(diff) ** 2) for diff in model_update['diffs']]
            model_data.initial_mse = np.mean(mses)
            model_data.decrypted_mse = model_data.initial_mse
            self.data_owner_service.send_mses(validators, model_data, mses)
            for i in range(1, self.n_iter + 1):
                last_mse = model_data.decrypted_mse
                model_data = self.training_cicle(model_data, i)
                if self.has_converged(model_data.decrypted_mse, last_mse) and (i % self.n_iter_partial_res) == 0:
                    logging.info("BREAKING")
                    break
            logging.info('Fished iterations for model {} {}'.format(model_id, model_data))
            model_buyer_data = {
                'model': {
                    "status": "FINISHED",
                    'weights': serialize(model_data.model.weights, self.encryption_service, model_data.public_key),
                    'id': model_data.model_id
                }
            }
            self.model_buyer_connector.send_result(model_buyer_data)
        except Exception as e:
            logging.error(e)
            self.send_error_to_model_buyer(model_id)

    def init_contract(self, model_id, model_buyer_addresss, trainers, validators):
        """
        Receives a data to add the participants in the smart contract and init new model to train
        :param model_id:
        :param model_buyer_addresss:
        :param trainers:
        :param validators:
        :return:
        """
        logging.info("Init contract")
        trainers_address = [trainer.address for trainer in trainers]
        validators_address = [validator.address for validator in validators]
        [self.smart_contract.set_data_owner(do_address) for do_address in (trainers_address + validators_address)]
        self.smart_contract.set_federated_aggregator(self.eth_address)
        self.smart_contract.set_model_buyer(model_buyer_addresss)
        self.smart_contract.new_model(model_id, validators_address, trainers_address, model_buyer_addresss)

    def split_data_owners(self, linked_data_owners):
        """
        Receives a list of data owners linked to a model_id. Returns two lists.
        The firsts contains the data owners that will participate training the model.
        The second contains the ones that will validate the training.
        :param linked_data_owners:
        :return: Two lists of data owners. The trainers and the validators-
        """
        split_index = int(len(linked_data_owners) * self.split_coefficient)
        copy = linked_data_owners[:]
        random.shuffle(copy)
        local_trainers, validators = copy[:split_index + 1], copy[split_index + 1:] or []
        logging.info("LocalTrainers: {}".format(list(map(lambda x: x.id, local_trainers))))
        logging.info("Validators: {}".format(list(map(lambda x: x.id, validators))))
        return local_trainers, validators

    @staticmethod
    def validate_linked_data_owners(linked_data_owners, model_id):
        if len(linked_data_owners) == 0:
            logging.error("Invalid Number of linked data owner")
            raise EmptyValidatorsException(model_id)

    def send_error_to_model_buyer(self, model_id):
        """
        TODO enumerate errors
        :param model_id:
        :return:
        """
        error_data = {
            "model": {
                'model_id': model_id,
                'status': "ERROR"
            }
        }
        self.model_buyer_connector.send_result(error_data)

    def has_converged(self, current_mse, last_mse):
        return last_mse is not None and current_mse is not None and self.loss_improvement(last_mse, current_mse) < 0.001

    @staticmethod
    def loss_improvement(last_mse, current_mse):
        return abs((last_mse - current_mse) / last_mse)

    def initialize_global_model(self, data):
        weights = self.encryption_service.get_deserialized_collection(data["weights"]) if self.active_encryption else \
            data["weights"]
        model = ModelFactory.get_model(data['model_type'])(weights=np.asarray(weights))
        # model.set_weights(np.asarray(weights))
        return model

    def training_cicle(self, model_data, i):
        gradients, local_trainers = self.data_owner_service.get_gradients(model_data)
        partial_models = [self.partial_update_model(deepcopy(model_data), gradients, local_trainers, i)
                          for i in range(len(local_trainers))]
        model_data.model.weights, avg_gradient = self.update_model(model_data, gradients)
        logging.info("Done updating model")
        if (i % self.n_iter_partial_res) == 0:
            logging.info("Calculating mses")
            diffs = self.data_owner_service.get_model_metrics_from_validators(model_data)
            partial_diffs = self.data_owner_service.get_partial_model_metrics_from_validators(partial_models,
                                                                                              model_data)
            logging.info("Sending partial results")
            model_update = self.send_partial_result_to_model_buyer(model_data, diffs, partial_diffs)
            mses = [np.mean(np.asarray(diff) ** 2) for diff in model_update['diffs']]
            model_data.decrypted_mse = np.mean(mses)
            model_data.mse = model_data.decrypted_mse
            self.data_owner_service.send_mses(model_data.validators, model_data, mses)
            model_data.model.weights = model_update['weights']
        self.data_owner_service.send_avg_gradient(avg_gradient, model_data)
        return model_data

    def update_model(self, model_data, gradients):
        """
        Updates the global model by performing a gradient descent step in the direction of the avg. gradient calculated
        from the gradients received from the local trainers.
        :param model_data: a wrapper that contains all the data related to a certain model in training.
        :param gradients: a list of gradients to be averaged.
        :return: the model updated after a step of gradient descent.
        """
        logging.info("Updating global model")
        avg_gradient = self.federated_averaging(gradients, model_data)
        model_data.model.gradient_step(avg_gradient, model_data.gradient_step)
        return model_data.model.weights, avg_gradient

    def partial_update_model(self, model_data, gradients, trainers, filtered_index):
        """
        Performs the same operation as the update_model method, but leaves out of the update one of the local trainers
        and its corresponding gradient. By doing that we can obtain a model that shows how better would have been to
        leave that local trainer out of the training.
        :param model_data: a wrapper that contains all the data related to a certain model in training.
        :param gradients: a list of gradients to be averaged.
        :param trainers: the list of the local trainers training local models and sending their gradients.
        :param filtered_index: the filtered local trainer and corresponding gradient.
        :return: a dictionary of models trained leaving different local trainers out. The filtered local trainers are
        the keys of the dictionary.
        """
        logging.info("Updating partial model for index {}".format(filtered_index))
        gradients = np.delete(gradients, filtered_index, 0)
        trainer = trainers[filtered_index]
        avg_gradient = self.federated_averaging(gradients, model_data)
        model_data.model.gradient_step(avg_gradient, model_data.gradient_step)
        return trainer, model_data.model.weights

    def send_partial_result_to_model_buyer(self, model_data, diffs, partial_diffs={}, first_update=False):
        for trainer in partial_diffs:
            partial_diffs[trainer] = [serialize(diff, self.encryption_service, model_data.public_key) for diff in
                                      partial_diffs[trainer]]
        weights = model_data.model.weights
        partial_result = {
            'first_update': first_update,
            'model': {
                'weights': serialize(weights, self.encryption_service, model_data.public_key),
                'id': model_data.model_id,
                'status': model_data.model_status,
                'type': model_data.model_type
            }, 'metrics': {
                'initial_mse': model_data.initial_mse,
                'mse': model_data.mse,
                'partial_diffs': partial_diffs,
                'diffs': [serialize(diff, self.encryption_service, model_data.public_key) for diff in diffs]
            }
        }
        model_update = self.model_buyer_connector.send_partial_result(partial_result)
        model_update['weights'] = deserialize(model_update['weights'], self.encryption_service, model_data.public_key)
        return model_update

    def federated_learning_wrapper(self, data):
        return self.federated_learning(data)

    def federated_averaging(self, updates, model_data):
        """
        Sum all de partial updates and
        :param model_data:
        :param updates:
        :return:
        """
        logging.info("Federated averaging")
        average = reduce(sum_collection, updates) / len(model_data.local_trainers)
        return average

    def send_prediction_to_buyer(self, data):
        """
        :param data:
        :return:
        """
        self.model_buyer_connector.send_encrypted_prediction(self.global_models[data["model_id"]], data)

    def send_prediction_to_data_owner(self, encrypted_prediction):
        model = self.global_models[encrypted_prediction["model_id"]]
        self.data_owner_service.send_encrypted_prediction(model=model, encrypted_prediction=encrypted_prediction)

    def serialize_if_activated(self, value):
        return self.encryption_service.get_serialized_encrypted_value(value) if self.config[
            "ACTIVE_ENCRYPTION"] else value

    def are_valid(self, model_id, mse, initial_mse, partial_MSEs, public_key):
        self.encryption_service.set_public_key(public_key)
        original_partial_MSEs = self.global_models[model_id].partial_MSEs
        encrypted_mse = self.serialize_if_activated(mse)
        encrypted_partial_MSEs = dict([(data_owner, self.serialize_if_activated(partial_MSE))
                                       for data_owner, partial_MSE in partial_MSEs.items()])
        mse_validated = self.compare_mses(encrypted_mse, self.global_models[model_id].mse)
        partial_MSEs_checks_out = self.compare_partial_mses(encrypted_partial_MSEs, original_partial_MSEs)
        return mse_validated and partial_MSEs_checks_out

    def compare_mses(self, mse_from_model_buyer, mse_from_validators):
        return mse_from_model_buyer == mse_from_validators

    def compare_partial_mses(self, mses_from_model_buyer, mses_from_validators):
        for data_owner in mses_from_model_buyer:
            if mses_from_model_buyer[data_owner] != mses_from_validators[data_owner]:
                return False
        return True

    def calculate_contributions(self, model_id, mse, initial_mse, partial_MSEs):
        improvement = max([(initial_mse - mse) / initial_mse, 0])
        contributions = {}
        for data_owner in partial_MSEs:
            owner_improvement = initial_mse - partial_MSEs[data_owner]
            contributions[data_owner] = owner_improvement if owner_improvement >= 0 else 0
        contributions_sum = sum(contributions.values())
        for data_owner in partial_MSEs:
            contributions[data_owner] = (contributions[data_owner]) / contributions_sum
        return {'model_id': model_id, 'improvement': improvement, 'contributions': contributions}
        # return model_id, improvement, contributions
