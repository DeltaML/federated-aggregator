import logging
import requests


class ModelBuyerConnector:

    def __init__(self, host, port):
        self.remote_address = host
        self.model_buyer_port = port

    def send_result(self, result):
        try:

            url = "http://{}:{}/models/{}".format(self.remote_address, self.model_buyer_port, result['model']["id"])
            payload = result
            # TODO Temporal fix
            payload["first_update"] = False
            logging.info("url {} payload {}".format(url, payload))
            response = requests.put(url, json=payload)
            response.raise_for_status()
            logging.info("Response {}".format(response.json()))
        except Exception as e:
            logging.error(e)

    def send_partial_result(self, result):
        url = "http://{}:{}/models/{}".format(self.remote_address, self.model_buyer_port, result['model']['id'])
        payload = result
        logging.info("url {} payload {}".format(url, payload))
        response = requests.patch(url, json=payload)
        response.raise_for_status()
        logging.info("Response {}".format(response.json()))
        return  response.json()

    def send_encrypted_prediction(self, model, encrypted_prediction):
        """

        :param model:
        :param encrypted_prediction:
        :return:
        """
        url = "http://{}:{}/transform".format(model.remote_address, self.model_buyer_port, model.model_id)
        payload = encrypted_prediction
        logging.info("Url {} payload {}".format(url, payload))
        response = requests.post(url, json=encrypted_prediction)
        response.raise_for_status()
        logging.info("Response {}".format(response.json()))
