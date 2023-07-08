import argparse
import os
import kserve

from insightfaceapp import InsightFaceModel

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
DEFAULT_RUNTIME_MODEL_NAME = os.getenv('DEFAULT_RUNTIME_MODEL_NAME')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
parser.add_argument('--runtime_model_name', default=DEFAULT_RUNTIME_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = InsightFaceModel(args.model_name, args.runtime_model_name)
    model.load()
    kserve.ModelServer().start([model])
