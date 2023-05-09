import argparse
import kserve
from typing import Dict
import os

class NoOpsModel(kserve.Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name

    def load(self):
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {"instances": [
        #   {
        #     "image_bytes": {
        #         "b64": "<b64-encoded>",
        #     },
        #     "key": "somekeys",
        #   },
        # ]}
        data = inputs[0]["image_bytes"]["b64"]
        key = inputs[0]["key"]
        # this model do no operations
        return {"predictions": [{
            "image_bytes": {
                "b64": data,
            },
            "type": "noops",
            "key": key,
        }]}

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = NoOpsModel(args.model_name)
    model.load()
    kserve.ModelServer().start([model])
