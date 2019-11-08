import json
import logging

import numpy as np
import requests

from commons.operations_utils.functions import serialize, deserialize
from commons.decorators.decorators import optimized_collection_response, normalize_optimized_collection_argument
from commons.utils.async_thread_pool_executor import AsyncThreadPoolExecutor
from federated_aggregator.utils.decorators import deserialize_encrypted_server_data, serialize_encrypted_server_gradient, deserialize_encrypted_server_data_2


class DataOwnerConnector:

    def __init__(self, data_owner_port, encryption_service, active_encryption):
        self.data_owner_port = data_owner_port
        self.async_thread_pool = AsyncThreadPoolExecutor()
        self.encryption_service = encryption_service
        self.active_encryption = active_encryption

    def send_gradient_to_data_owners(self, data_owners, gradient, model_id, public_key):

        args = [self._build_data(data_owner, gradient, model_id, public_key) for data_owner in data_owners]
        self.async_thread_pool.run(executable=self._send_gradient, args=args)

    #@optimized_dict_collection_response(optimization=np.asarray, active=True)
    def get_gradient_from_data_owners(self, model_data):
        args = [
            (trainer, model_data.model_type, model_data.model.weights, model_data.model_id, model_data.public_key)
            for trainer in model_data.local_trainers
        ]
        return self.async_thread_pool.run(executable=self._get_update_from_data_owner, args=args)

    @optimized_collection_response(optimization=np.asarray, active=True)
    def get_data_owners_model(self, model_data):
        args = [
            "http://{}:{}/model".format(trainer.host, self.data_owner_port)
            for trainer in model_data.local_trainers
        ]
        results = self.async_thread_pool.run(executable=self._send_get_request_to_data_owner, args=args)
        return [result for result in results]

    def send_requirements_to_data_owners(self, data_owners, data):
        args = [
            ("http://{}:{}/trainings".format(data_owner.host, self.data_owner_port), data)
            for data_owner in data_owners
        ]
        self.async_thread_pool.run(executable=self._send_post_request_to_data_owner, args=args)

    @optimized_collection_response(optimization=np.asarray, active=True)
    def get_linked_data_owners(self, data_owners, model_id):
        args = [
            "http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id)
            for data_owner in data_owners
        ]
        results = self.async_thread_pool.run(executable=self._send_get_request_to_data_owner, args=args)
        return [result for result in results]

    def send_mses(self, validators, model_data, mses, role):
        args = [
            (
            "http://{}:{}/trainings/{}/metrics".format(validators[i].host, self.data_owner_port, model_data.model_id), {'mse': mses[i], 'role': role})
            for i in range(len(validators))
        ]
        self.async_thread_pool.run(executable=self._send_put_request_to_data_owner, args=args)

    def get_model_metrics_from_validators(self, validators, model_data, weights=None):
        model = weights if weights is not None else model_data.model.weights
        data = {'model': serialize(model, self.encryption_service, model_data.public_key),
                'model_type': model_data.model_type,
                'model_id': model_data.model_id,
                'public_key': model_data.public_key
                }
        args = [
            ("http://{}:{}/trainings/{}/metrics".format(validator.host, self.data_owner_port, model_data.model_id), data)
            for validator in validators
        ]
        results = self.async_thread_pool.run(executable=self._send_post_request_to_data_owner, args=args)
        results = [result['diff'] for result in results]
        results = [deserialize(result, self.encryption_service, model_data.public_key) for result in results]
        return results

    @deserialize_encrypted_server_data()
    def _get_update_from_data_owner(self, data):
        """
        :param data:
        :return:
        """
        data_owner, model_type, weights, model_id, public_key = data
        url = "http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id)
        payload = {"model_type": model_type, "weights": self.encryption_service.get_serialized_collection(weights) if self.active_encryption else weights, "public_key": public_key}
        logging.info("Url: {}".format(url))
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info("response {}".format(response))
        return response.json()

    @serialize_encrypted_server_gradient(schema=json.dumps)
    def _send_gradient(self, data):
        """
        Replace with parallel
        :param data:
        :return:
        """
        url, payload = data
        logging.info("Url: {} ".format(url))
        response = requests.put(url, json=payload)
        response.raise_for_status()
        logging.info("response {}".format(response))

    def send_result_to_data_owners(self, model_id, contribs, data_owners):
        args = [
            ("http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id), {'contribs': contribs})
            for data_owner in data_owners
        ]
        self.async_thread_pool.run(executable=self._send_patch_request_to_data_owner, args=args)

    @deserialize_encrypted_server_data_2()
    def _send_get_request_to_data_owner(self, url):
        logging.info("Url: {} ".format(url))
        response = requests.get(url).json()
        response.raise_for_status()
        logging.info("Response {}".format(response))
        return response

    @normalize_optimized_collection_argument(active=True)
    def _build_data(self, data_owner, gradient, model_id, public_key):
        return "http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id), {"gradient": gradient, "public_key": public_key}

    @staticmethod
    def _send_post_request_to_data_owner(data):
        url, payload = data
        logging.info("Url: {} ".format(url))
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        logging.info("Response: {} ".format(response))
        return response.json()

    @staticmethod
    def _send_put_request_to_data_owner(data):
        url, payload = data
        logging.info("Url: {} ".format(url))
        response = requests.put(url, json=payload, timeout=None)
        response.raise_for_status()
        logging.info("Response: {} ".format(response))
        return response.json()

    @staticmethod
    def _send_patch_request_to_data_owner(data):
        url, payload = data
        logging.info("Url: {} ".format(url))
        response = requests.patch(url, json=payload, timeout=None)
        response.raise_for_status()
        logging.info("Response: {} ".format(response))
        return response.json()

    def send_encrypted_prediction(self, data_owner, encrypted_prediction):
        """
        {'model_id': model_id,
         'prediction_id': prediction_id,
         'encrypted_prediction': Data Owner encrypted prediction,
         'public_key': Data Owner PK
         }
        :param data_owner:
        :param encrypted_prediction:
        :return:
        """
        url = "http://{}:{}/predictions/{}".format(data_owner.host, self.data_owner_port,
                                                   encrypted_prediction["prediction_id"])
        payload = encrypted_prediction
        logging.info("Url {} payload".format(url))
        response = requests.patch(url, json=payload)
        response.raise_for_status()
        logging.info("Response {}".format(response))
        return response
