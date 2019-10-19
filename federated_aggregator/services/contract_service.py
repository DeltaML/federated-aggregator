import logging
from commons.web3.delta_contracts import FederatedAggregatorContract


class ContractService:

    def __init__(self, w3_service, contract_address, fa_account):
        self.w3_service = w3_service
        self.smart_contract = FederatedAggregatorContract(contract=w3_service.build_contract(address=contract_address),
                                                          address=fa_account)
        self.fa_account = fa_account

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
        [self.smart_contract.set_data_owner(do_address) for do_address in (trainers_address + validators_address)]
        self.smart_contract.set_federated_aggregator(self.fa_account)
        self.smart_contract.set_model_buyer(global_model.model_buyer.address)
        # Create model into smart contract
        self.smart_contract.new_model(global_model.model_id, validators_address, trainers_address,
                                      global_model.model_buyer.address)
        # Save initial mse
        self.smart_contract.save_mse(global_model.model_id, global_model.initial_mse, 0)

    def save_mse(self, model_id, mse, iteration):
        """

        :param model_id:
        :param mse:
        :param iteration:
        :return:
        """

        logging.info("save_mse contract")
        self.smart_contract.save_mse(model_id, int(mse), iteration)

    def save_partial_mse(self, model_id, mse, trainer, iteration):
        """

        :param model_id:
        :param mse:
        :param trainer:
        :param iteration:
        :return:
        """
        logging.info("save_partial_mse contract")
        logging.info("Saving partial_mse model_id:{}, mse:{}, trainer_addr:{}, iter:{}".format(model_id, int(mse), trainer, iteration))
        self.smart_contract.save_partial_mse(model_id, int(mse), trainer, iteration)
