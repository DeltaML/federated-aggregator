from flask_restplus import Resource, Namespace
api = Namespace('helper', description='Helper related operations')


@api.route('/ping', endpoint='helper_resources_ep')
class HelperResources(Resource):
    @api.doc('Ping')
    def post(self):
        return 200
