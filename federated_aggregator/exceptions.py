class ExceptionMessageConstants:
    EMPTY_VALIDATORS = "Not found validators in training"


class FederatedTrainerException(Exception):
    def __init__(self, msg, status_code=400):
        # Call the base class constructor with the parameters it needs
        super().__init__(msg)

        # Now for your custom code...
        self.status_code = status_code


class EmptyValidatorsException(FederatedTrainerException):
    def __init__(self, data):
        super().__init__("{} {}".format(ExceptionMessageConstants.EMPTY_VALIDATORS, data))
