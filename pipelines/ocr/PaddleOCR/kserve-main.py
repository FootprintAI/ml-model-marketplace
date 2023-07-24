import argparse
import os
import kserve

from kserve_paddleocr import PaddleOCRModel

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
parser.add_argument('--lang', default=DEFAULT_LANGUAGE)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = PaddleOCRModel(args.model_name, args.lang)
    model.load()
    kserve.ModelServer().start([model])
