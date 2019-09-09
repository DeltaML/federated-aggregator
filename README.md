# Federated learning - Federated Aggregator

[![Build Status](https://travis-ci.com/DeltaML/federated-aggregator.svg?branch=master)](https://travis-ci.com/DeltaML/federated-aggregator)
[![Coverage Status](https://coveralls.io/repos/github/DeltaML/federated-aggregator/badge.svg?branch=master)](https://coveralls.io/github/DeltaML/federated-aggregator?branch=master)

Repository that contains a Proof of Concept for the implementation of a Federated Learning framework.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


- [python3](https://www.python.org/download/releases/3.0/)
- [docker](https://www.docker.com/)
- [docker-compose](https://docs.docker.com/compose/)


## Installing

A step by step series that tell you how to get a development env running

```
git clone git@github.com:DeltaML/federated_aggregator.git
cd federated_aggregator/
python3 -m venv venv
source venv/bin/activate
pip install -r federated_aggregator/requirements.txt
```

## Run

### Using command line
``` bash
    gunicorn -b "0.0.0.0:8080" --chdir federated_aggregator/ wsgi:app --preload
``` 


### Using Docker
``` bash
    docker build -t federated-learning-federated-trainer --rm -f federated-trainer/Dockerfile
    docker run --rm -it -p 8080:8080federated-learning-federated-trainer
``` 


### Using Pycharm

	Script Path: .../federated_aggregator/virtualenv/bin/gunicorn
	Parameters: -b "0.0.0.0:8080" wsgi:app --preload
	Working directory: ../federated_aggregator


## Usage 
 
### Register new data owner

``` bash
curl -v -H "Content-Type: application/json" -X POST "http://localhost:8080/dataowner"
```

### Get data owners registered

``` bash
curl -v -H "Content-Type: application/json" -X GET "http://localhost:8080/dataowner"
```

### Train model

``` bash
curl -v -H "Content-Type: application/json" -X POST -d '{"type": "LINEAR_REGRESSION", "call_back_endpoint": "URL_MODEL_BUYER", "call_back_port": 9090,"public_key": "XXXXXXXXXXXXXXXX"}' "http://localhost:8080/model"
```


## Configuration

``` python3
ACTIVE_ENCRYPTION = False
N_ITER = 100 # El numero de iteraciones aceptables utilizando PheEncryption por ahora es 4
DATA_OWNER_PORT = 5000

```

### Configuration details
- ACTIVE_ENCRYPTION: __TODO__
- N_ITER: __TODO__
- DATA_OWNER_PORT: __TODO__


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/DeltaML/federated-learning-poc/tags). 

## Authors

* **Fabrizio Graffe** - *Dev* - [GFibrizo](https://github.com/GFibrizo)
* **Agustin Rojas** - *Dev* - [agrojas](https://github.com/agrojas)

See also the list of [contributors](https://github.com/DeltaML/federated-learning-poc/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

