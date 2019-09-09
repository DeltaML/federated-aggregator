import numpy as np


def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)


def sum_collection(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return x + y


def deserialize(collection, encryption_service, public_key):
    """
    Deserializes a collection if the encryption service is on.
    Returns the collection as a numpy array.
    :param collection: a collection of numbers or serialized encrypted numbers if the encryption_service is on.
    :param encryption_service: the encryption service.
    :param public_key: the public key for deserializing numbers.
    :return: A numpy array with the content deserialized if the encryption service is on.
    """
    encryption_service.set_public_key(public_key)
    collection = encryption_service.get_deserialized_collection(collection) if encryption_service.is_active else collection
    return np.asarray(collection)


def serialize(collection, encryption_service, public_key):
    """
    Serializes a collection if the encryption service is on.
    Returns the collection as a list.
    :param collection: a collection of numbers or encrypted numbers if the encryption_service is on.
    :param encryption_service: the encryption service.
    :param public_key: the public key for serializing numbers.
    :return: A list with the content serialized if the encryption service is on.
    """
    encryption_service.set_public_key(public_key)
    collection = collection.tolist()
    return encryption_service.get_serialized_collection(collection) if encryption_service.is_active else collection
