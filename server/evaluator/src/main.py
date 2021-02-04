import tornado.ioloop
import tornado.web

import json

#from optimization import fullParametrizedFitnessFunction
from optimization2 import evaluateSingleFLCSimulation

globalkey = '_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__'
globalkey = 'FFFF'

class ApiHandler(tornado.web.RequestHandler):

    def prepare(self):
        #print('preparing')
        #print(self.request.body)
        if 'Content-Type' in self.request.headers:
            if self.request.headers['Content-Type'] == 'application/x-json':
                self.args = json.loads(self.request.body)

    async def post(self, key):
        #print('posting')
        if key == globalkey:
            pass
        
        result = evaluateSingleFLCSimulation(self.args)
        self.write(result)
        pass


class EchoHandler(tornado.web.RequestHandler):

    def prepare(self):
        print('preparing')
        if 'Content-Type' in self.request.headers:
            if self.request.headers['Content-Type'] == 'application/x-json':
                self.args = json.loads(self.request.body)

    async def post(self, key):
        print('posting')
        if key == globalkey:
            pass
        
        result = (self.args)
        self.write(result)
        pass

    async def get(self, key):
        print('getting')
        if key == globalkey:
            pass
        
        result = 'Hello world'
        self.write(result)
        pass    

def make_app(**kwargs):
    return tornado.web.Application([
        (r"/api/evaluator/([0-9a-zA-Z\-]*)", ApiHandler),
        (r"/", EchoHandler),
    ], **kwargs)

if __name__ == "__main__":
    settings = {
    }

    app = make_app(debug=True, **settings)
    app.listen(80)
    tornado.ioloop.IOLoop.current().start()
