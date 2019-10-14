class GlobalModel:
    def __init__(self,
                 buyer_id,
                 buyer_host,
                 model_id,
                 model_type,
                 model_status,
                 data_owners,
                 local_trainers,
                 validators,
                 model,
                 initial_mse,
                 mse,
                 public_key,
                 partial_MSEs,
                 step):
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
        self.buyer_id = buyer_id
        self.model_id = model_id
        self.buyer_host = buyer_host
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
        self.decrypted_mse = None
        self.gradient_step = step
