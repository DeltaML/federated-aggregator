import logging

from commons.utils.singleton import Singleton
from commons.web3.delta_contracts import FederatedAggregatorContract
from commons.web3.web3_service import Web3Service


class ContractService(metaclass=Singleton):

    def __init__(self):
        self.w3_service = None
        self.default_contract_address = None
        self.contract_address = None
        self.fa_account = None

    def init(self, config):
        self.w3_service = Web3Service(config["ETH_URL"])
        self.default_contract_address = config["CONTRACT_ADDRESS"]
        self.fa_account = config["FEDERATED_AGGREGATOR_ADDRESS"]
        self.contract_address = self.default_contract_address

    def set_contract_address(self, address):
        self.contract_address = address

    def get_contract_data(self):
        return {'address': self.contract_address}

    def set_federated_aggregator_address(self, address):
        self.fa_account = address

    def get_federated_aggregator_address(self):
        return {'address': self.fa_account}

    def build_contract_api(self):
        """

        :param account:
        :return:
        """
        return FederatedAggregatorContract(contract=self.w3_service.build_contract(address=self.contract_address),
                                           address=self.fa_account)

    def init_contract(self, global_model):
        """
        Receives a data to add the participants in the smart contract and init new model to train
        :param global_model:
        :return:
        """
        logging.info("Init contract")
        trainers_address = [trainer.address for trainer in global_model.local_trainers]
        validators_address = [validator.address for validator in global_model.validators]
        # Add actors
        smart_contract = self.build_contract_api()
        [smart_contract.set_data_owner(do_address) for do_address in (trainers_address + validators_address)]
        smart_contract.set_federated_aggregator(self.fa_account)
        smart_contract.set_model_buyer(global_model.model_buyer.address)
        # Create model into smart contract
        smart_contract.new_model(global_model.model_id, validators_address, trainers_address,
                                 global_model.model_buyer.address)
        # Save initial mse
        smart_contract.save_mse(global_model.model_id, global_model.initial_mse, 0)

    def save_mse(self, model_id, mse, iteration):
        """

        :param model_id:
        :param mse:
        :param iteration:
        :return:
        """

        logging.info("save_mse contract")
        self.build_contract_api().save_mse(model_id, int(mse), iteration)

    def save_partial_mse(self, model_id, mse, trainer, iteration):
        """

        :param model_id:
        :param mse:
        :param trainer:
        :param iteration:
        :return:
        """
        logging.info(
            "Saving partial_mse model_id:{}, mse:{}, trainer_addr:{}, iter:{}".format(model_id, int(mse), trainer,
                                                                                      iteration))
        self.build_contract_api().save_partial_mse(model_id, int(mse), trainer, iteration)
