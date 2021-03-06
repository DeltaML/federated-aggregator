
class ModelBuyer:
    def __init__(self, data):
        self.id = data["model_buyer_id"]
        self.host = data["model_buyer_host"]
        self.address = data["model_buyer_address"]


class Payment:
    def __init__(self, data):
        self.unit = data["unit"]
        self.value = data["value"]


class PaymentsData:
    def __init__(self, data):
        self.pay_for_model = Payment(data["pay_for_model"])


class GlobalModel:
    DEFAULT_INITIAL_MSE = 1000

    def __init__(self,
                 model_buyer_data,
                 model_id,
                 model_type,
                 model_status,
                 model,
                 public_key,
                 step,
                 payments_data,
                 data_owners=[],
                 local_trainers=[],
                 validators=[],
                 partial_MSEs=None,
                 initial_mse=DEFAULT_INITIAL_MSE,
                 mse=None
                 ):
        """

        :param buyer_id: String
        :param buyer_host:
        :param model_id: String
        :param public_key: String
        :param model_type: String
        :param data_owners: List[]
        :param local_trainers: List[]
        :param validators: List[]
        :param model: LinearRegression
        """
        self.model_id = model_id
        self.model_buyer = ModelBuyer(model_buyer_data)
        self.model_type = model_type
        self.model_status = model_status
        self.data_owners = data_owners
        self.local_trainers = local_trainers
        self.validators = validators
        self.model = model
        self.initial_mse = initial_mse
        self.mse = mse
        self.partial_MSEs = partial_MSEs
        self.public_key = public_key
        self.gradient_step = step
        self.payments_data = PaymentsData(payments_data)
        self.decrypted_mse = None

