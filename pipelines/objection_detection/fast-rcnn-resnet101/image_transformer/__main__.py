import argparse
import json
import kserve
import os
from .image_transformer import ImageTransformer
from .transformer_model_repository import TransformerModelRepository

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)

args, _ = parser.parse_known_args()

#CONFIG_PATH = "/mnt/models/config/config.properties"


def parse_config():
    #separator = "="
    #keys = {}

    #with open(CONFIG_PATH) as f:

    #    for line in f:
    #        if separator in line:

    #            # Find the name and value by splitting the string
    #            name, value = line.split(separator, 1)

    #            # Assign key value pair to dict
    #            # strip() removes white space from the ends of strings
    #            keys[name.strip()] = value.strip()

    #keys["model_snapshot"] = json.loads(keys["model_snapshot"])

    #models = keys["model_snapshot"]["models"]
    #model_names = []

    ## Get all the model_names
    #for model, value in models.items():
    #    model_names.append(model)
    #if not model_names:
    #    model_names = [DEFAULT_MODEL_NAME]
    model_names = [DEFAULT_MODEL_NAME]
    print(f"Wrapper : Model names {model_names}")
    return model_names


if __name__ == "__main__":
    model_names = parse_config()
    models = []
    for model_name in model_names:
        transformer = ImageTransformer(model_name, predictor_host=args.predictor_host)
        transformer.load()
        models.append(transformer)
    kserve.ModelServer(
        registered_models=TransformerModelRepository(args.predictor_host)
    ).start(models=models)
