

@app.route('/prediction', methods=['POST'])
def post_prediction():
    data = request.get_json()
    logging.info("Data {}".format(data))
    federated_aggregator.send_prediction_to_buyer(data)
    return jsonify(200), 200


@app.route('/prediction/<prediction_id>', methods=['PATCH'])
def patch_prediction(prediction_id):
    data = request.get_json()
    logging.info("Data {}".format(data))
    federated_aggregator.send_prediction_to_data_owner(data)
    return jsonify("pong")
