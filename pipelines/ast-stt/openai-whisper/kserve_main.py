import argparse
import os
import kserve

from whisper_kserve import WhisperModel

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = WhisperModel(args.model_name)
    model.load()
    kserve.ModelServer().start([model])
