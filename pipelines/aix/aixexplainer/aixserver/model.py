# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import asyncio
import base64
import io
import logging
import kserve
import numpy as np
from aix360.algorithms.lime import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from aix360.algorithms.lime import LimeTextExplainer
import nest_asyncio
nest_asyncio.apply()

from skimage.color import label2rgb # since the code wants color images

def base64decode(s:str):
    import base64
    import cv2
    import numpy as np

    jpg_original = base64.b64decode(s)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    im = cv2.imdecode(jpg_as_np, cv2.IMREAD_UNCHANGED)
    return im

def base64encode(im) -> str:
    import base64
    import cv2

    im_encode = cv2.imencode('.jpg', im)[1]
    return base64.b64encode(im_encode).decode('utf-8')

def normalize(im):
    import cv2

    return cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)

def resize(im, size=(240,320)):
    import cv2

    h, w, channels = im.shape
    aspect_ratio = h/w

    target_height, target_width = size
    golden_ratio = target_height / target_width
    new_height, new_width = size
    if aspect_ratio > golden_ratio:
        new_width = int(new_height / aspect_ratio)
    else:
        new_height = int(new_width * aspect_ratio)

    return cv2.resize(im, (new_width, new_height), interpolation = cv2.INTER_NEAREST)


class AIXModel(kserve.Model):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, predictor_host: str, segm_alg: str, num_samples: str,
                 top_labels: str, min_weight: str, positive_only: str, explainer_type: str):
        super().__init__(name)
        logging.info('AIXModel, segm_alg:{}, positive_only:{},  explainer_type:{}, toplabel:{}'.format(segm_alg, positive_only, explainer_type, top_labels))
        self.name = name
        self.top_labels = int(top_labels)
        self.num_samples = int(num_samples)
        self.segmentation_alg = segm_alg
        self.predictor_host = predictor_host
        self.min_weight = float(min_weight)
        self.positive_only = (positive_only.lower() == "true") | (
            positive_only.lower() == "t")
        if str.lower(explainer_type) != "limeimages" and str.lower(explainer_type) != "limetexts":
            raise Exception("Invalid explainer type: %s" % explainer_type)
        self.explainer_type = explainer_type
        self.ready = False

    def load(self) -> bool:
        self.ready = True
        return self.ready

    def _predict(self, input_im):
        """ _predict send $input_im into the underlying predictor.
        Each input would be wrapped with {"image_bytes": {"b64": $b64str}, "key": $key}
        And the output from prediction is {"scores": $list_of_scores_for_each_class, "prediction": $predict_class, "key": $key}
        or {[$list_of_scores_for_each_class]}
        we use _wrap_numpyarr_to_predict_inputs to wrap the input

        """

        # NOTE(hsiny): some predictors may take differnet input shapes, e.g.
        # some would take a list of tensors instead of a b64 encoded image_byte
        # we may need custom encoding according to the predictor's metadata
        # see this for more details: https://github.com/kserve/kserve/issues/2304
        scoring_data = self._wrap_numpyarr_to_predict_inputs(input_im)

        loop = asyncio.get_running_loop()
        resp = loop.run_until_complete(self.predict(scoring_data))
        predictions = resp["predictions"]
        # output: "predictions": [
        #        {
        #            "scores": [1.47944235e-07, 3.65586068e-08, 0.796582818, 1.05895253e-07, 0.203416958, 3.8090274e-08],
        #            "prediction": 2,
        #            "key": "1"
        #        }
        #    ]
        #
        # or output: "predictions":[
        #        {
        #            [1.47944235e-07, 3.65586068e-08, 0.796582818, 1.05895253e-07, 0.203416958, 3.8090274e-08],
        #        }
        #]
        #for sample_inx in range(0, len(predictions)):
        #    print('{}:{}'.format(sample_inx, predictions[sample_inx]))

        class_preds = [ sample_against_all['scores'] if
                               'scores' in sample_against_all else
                               sample_against_all for
                               sample_against_all in predictions]
        return np.array(class_preds)

    def _wrap_numpyarr_to_predict_inputs(self, input_im: np.ndarray):
        instances = []
        index = 1;
        for slice_input_im in input_im:
            b64str = base64encode(slice_input_im)
            instances.append({"image_bytes": {"b64": b64str}, "key": "{}".format(index)})
            index = index + 1
        return {"instances": instances}


    def explain(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        instances = payload["instances"]
        try:
            top_labels = (int(payload["top_labels"])
                          if "top_labels" in payload else
                          self.top_labels)
            segmentation_alg = (payload["segmentation_alg"]
                                if "segmentation_alg" in payload else
                                self.segmentation_alg)
            num_samples = (int(payload["num_samples"])
                           if "num_samples" in payload else
                           self.num_samples)
            positive_only = ((payload["positive_only"].lower() == "true") | (payload["positive_only"].lower() == "t")
                             if "positive_only" in payload else
                             self.positive_only)
            min_weight = (float(payload['min_weight'])
                          if "min_weight" in payload else
                          self.min_weight)
            explaim_image_width = (float(payload['explain_image_width'])
                          if 'explain_image_width' in payload else
                          320)
            explaim_image_height = (float(payload['explain_image_height'])
                          if 'explain_image_height' in payload else
                          240)
        except Exception as err:
            raise Exception("Failed to specify parameters: %s", (err,))

        try:
            if str.lower(self.explainer_type) == "limeimages":
                input_im = self._get_instance_binary_inputs(instances[0], (explaim_image_height, explaim_image_width))
                inputs = np.array(input_im)
                logging.info(
                    "Calling explain on image of shape %s", (inputs.shape,))
            elif str.lower(self.explainer_type) == "limetexts":
                inputs = str(instances[0])
                logging.info("Calling explain on text %s", (len(inputs),))
        except Exception as err:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (err, instances))
        try:
            if str.lower(self.explainer_type) == "limeimages":
                explainer = LimeImageExplainer(verbose=False)
                segmenter = SegmentationAlgorithm(segmentation_alg, kernel_size=4,
                                                  max_dist=200, ratio=0.2)
                explanation = explainer.explain_instance(inputs,
                                                         classifier_fn=self._predict,
                                                         top_labels=top_labels,
                                                         hide_color=0,
                                                         num_samples=num_samples,
                                                         segmentation_fn=segmenter)

                explained_imageb64_list = []
                #logging.info("local-pred:{}".format(explanation.local_pred))
                #logging.info("local-exp:{}".format(explanation.local_exp))
                for i in range(0, top_labels):
                    temp, mask = explanation.get_image_and_mask(explanation.top_labels[i],
                                                                positive_only=positive_only,
                                                                num_features=10,
                                                                hide_rest=False,
                                                                min_weight=min_weight)
                    explained_imageb64_list.append({
                            "image_bytes": {
                                "b64": base64encode(normalize(label2rgb(mask,input_im, bg_label = 0)))
                            }
                        })

                return {"explanations": {
                    "type": "limeimages",
                    "explained_imageb64_list": explained_imageb64_list,
                    "top_labels": np.array(explanation.top_labels).astype(np.int32).tolist()
                }}
            elif str.lower(self.explainer_type) == "limetexts":
                # NOTE(hsiny): for limetexts route, we haven't test it yet.

                explainer = LimeTextExplainer(verbose=False)
                explaination = explainer.explain_instance(inputs,
                                                          classifier_fn=self._predict,
                                                          top_labels=top_labels)
                m = explaination.as_map()
                exp = {str(k): explaination.as_list(int(k))
                       for k, _ in m.items()}

                return {"explanations": exp}

        except Exception as err:
            raise Exception("Failed to explain %s" % err)

    def _get_instance_binary_inputs(self, first_instance, preferred_size):
        """ _get_instance_binary_inputs converts b64 encoded instance's
        imagebytes into numpy.array
        """

        if isinstance(first_instance, dict) and "image_bytes" in first_instance and "b64" in first_instance["image_bytes"]: # first_instance = {"image_bytes": {"b64":xxx}}
            logging.info("first instance is dict and has b64, coverting")
            return resize(base64decode(first_instance["image_bytes"]["b64"]), preferred_size)

        return first_instance
