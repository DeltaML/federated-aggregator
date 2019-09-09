import json
import logging

import numpy as np
import requests


from commons.decorators.decorators import optimized_collection_response, normalize_optimized_collection_argument, optimized_dict_collection_response
from commons.utils.async_thread_pool_executor import AsyncThreadPoolExecutor
from federated_aggregator.service.decorators import deserialize_encrypted_server_data, serialize_encrypted_server_gradient, deserialize_encrypted_server_data_2


class DataOwnerConnector:

    def __init__(self, data_owner_port, encryption_service, active_encryption):
        self.data_owner_port = data_owner_port
        self.async_thread_pool = AsyncThreadPoolExecutor()
        self.encryption_service = encryption_service
        self.active_encryption = active_encryption

    def send_gradient_to_data_owners(self, data_owners, gradient, model_id):
        args = [self._build_data(data_owner, gradient, model_id) for data_owner in data_owners]
        self.async_thread_pool.run(executable=self._send_gradient, args=args)

    #@optimized_dict_collection_response(optimization=np.asarray, active=True)
    def get_gradient_from_data_owners(self, model_data):
        args = [
            (trainer, model_data.model_type, model_data.model.weights, model_data.model_id)
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

    @optimized_collection_response(optimization=np.asarray, active=True)
    def send_requirements_to_data_owners(self, data_owners, data):
        args = [
            ("http://{}:{}/trainings".format(data_owner.host, self.data_owner_port), data)
            for data_owner in data_owners
        ]
        results = self.async_thread_pool.run(executable=self._send_post_request_to_data_owner, args=args)
        return [result for result in results]

    @optimized_collection_response(optimization=np.asarray, active=True)
    def get_model_metrics_from_validators(self, validators, model_data, weights=None):
        model = weights if weights is not None else model_data.model.weights
        logging.info(model)
        data = {'model': model.tolist(),
                'model_type': model_data.model_type,
                'model_id': model_data.model_id,
                'public_key': model_data.public_key}
        args = [
            ("http://{}:{}/trainings/{}/metrics".format(validator.host, self.data_owner_port, model_data.model_id), data)
            for validator in validators
        ]
        results = self.async_thread_pool.run(executable=self._send_post_request_to_data_owner, args=args)
        return [result['mse'] for result in results]

    @deserialize_encrypted_server_data()
    def _get_update_from_data_owner(self, data):
        """
        :param data:
        :return:
        """
        data_owner, model_type, weights, model_id = data
        url = "http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id)
        payload = {"model_type": model_type, "weights": weights.tolist()}
        logging.info("Url: {} payload {}".format(url, payload))
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info("response {}".format(response.json()))
        return response.json()

    @serialize_encrypted_server_gradient(schema=json.dumps)
    def _send_gradient(self, data):
        """
        Replace with parallel
        :param data:
        :return:
        """
        url, payload = data
        logging.info("Url: {} payload {}".format(url, payload))
        response = requests.put(url, json=payload)
        response.raise_for_status()
        logging.info("response {}".format(response.json()))

    @deserialize_encrypted_server_data_2()
    def _send_get_request_to_data_owner(self, url):
        logging.info("Url: {} ".format(url))
        response = requests.get(url).json()
        response.raise_for_status()
        logging.info("Response {}".format(response))
        return response

    # ---
    def _build_data(self, data_owner, gradient, model_id):
        return "http://{}:{}/trainings/{}".format(data_owner.host, self.data_owner_port, model_id), {"gradient": gradient}

    @staticmethod
    def _send_post_request_to_data_owner(data):
        url, payload = data
        logging.info("Url: {} payload {}".format(url, payload))
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        logging.info("Response: {} ".format(response.json()))
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
        logging.info("Url {} payload".format(url, payload))
        response = requests.patch(url, json=payload)
        response.raise_for_status()
        logging.info("Response {}".format(response))
        return response
