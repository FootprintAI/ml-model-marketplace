import io
import base64
import json
import tornado
import numpy as np
from typing import List, Dict
from PIL import Image
import tensorflow as tf
import logging
import kserve
import json

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

def image_transform(instance):
    byte_array = base64.b64decode(instance['image_bytes']['b64'])
    image = Image.open(io.BytesIO(byte_array))
    a = np.asarray(image)
    tensor = tf.convert_to_tensor(a, dtype=tf.uint8)
    input_tensor = tf.expand_dims(tensor, axis=0)
    return input_tensor.numpy().tolist()

class ImageTransformer(kserve.Model):
    """ A class object for the data handling activities of Image Classification
    Task and returns a KServe compatible response.
    Args:
        kserve (class object): The Model class from the KServe
        module is passed here.
    """
    def __init__(self, name: str, predictor_host: str):
        """Initialize the model name, predictor host and the explainer host
        Args:
            name (str): Name of the model.
            predictor_host (str): The host in which the predictor runs.
        """
        super().__init__(name)
        self.predictor_host = predictor_host
        self.explainer_host = predictor_host
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        logging.info("EXPLAINER URL %s", self.explainer_host)
        self.timeout = 100

    # NOTE(hsiny): this transformer will take the first element of the
    # instances and sent to the the model served at back, then wrapped as list
    def preprocess(self, inputs: Dict) -> Dict:
        """Pre-process activity of the Image Input data.
        Args:
            inputs (Dict): KServe http request
        Returns:
            Dict: Returns the request input after converting it into a tensor
        """
        # limit one instance at a time
        return {'instances': image_transform(inputs['instances'][0])}

    def postprocess(self, inputs: Dict) -> List:
        """Post process function of Torchserve on the KServe side is
        written here.
        Args:
            inputs (Dict): The Dict of the inputs
        Returns:
            List: If a post process functionality is specified, it converts that into
            a list.
        """
        if "predictions" in inputs:
            predictions = inputs['predictions']
            for prediction in predictions:
                if 'type' not in prediction:
                    prediction['type'] = 'object-detection-coco17'
        return inputs

    async def explain(self, request: Dict) -> Dict:
        """Returns the captum explanations for the input request
        Args:
            request (Dict): http input request
        Raises:
            NotImplementedError: If the explainer host is not specified.
            tornado.web.HTTPError: if the response code is not 200.
        Returns:
            Dict: Returns a dictionary response of the captum explain
        """
        if self.explainer_host is None:
            raise NotImplementedError
        logging.info("Inside Image Transformer explain %s", EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name))
        response = await self._http_client.fetch(
            EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name),
            method='POST',
            request_timeout=self.timeout,
            body=json.dumps(request)
        )
        if response.code != 200:
            raise tornado.web.HTTPError(
                status_code=response.code,
                reason=response.body)
        return json.loads(response.body)
