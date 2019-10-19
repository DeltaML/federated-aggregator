class ExceptionMessageConstants:
    EMPTY_VALIDATORS = "Not found validators in training {}"
    INVALID_ABI_FORMAT = "Abi {} haven't valid format"


class FederatedTrainerException(Exception):
    def __init__(self, msg, status_code=400):
        # Call the base class constructor with the parameters it needs
        super().__init__(msg)

        # Now for your custom code...
        self.status_code = status_code


class EmptyValidatorsException(FederatedTrainerException):
    def __init__(self, data):
        super().__init__(ExceptionMessageConstants.EMPTY_VALIDATORS.format(data))
