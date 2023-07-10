import cv2
import numpy as np
import insightface
import kserve
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from typing import Dict


# imread read image and converts it into GRB
def imread(filepath:str):
    import cv2

    im = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

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
    return base64.b64encode(im_encode)

class InsightFaceModel(kserve.Model):
    def __init__(self, name: str, runtime_model_name: str):
        super().__init__(name)
        self.name = name
        self.runtime_model_name = runtime_model_name
        self.app = None

    def load(self):
        self.app = FaceAnalysis(name=self.runtime_model_name, root='/mnt', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # marked as ready
        self.ready = True

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {
        #   "instances": [
        #     {
        #       "image_bytes": {
        #           "b64": "<b64-encoded>",
        #       },
        #       "key": "somekeys",
        #     },
        #   ],
        # }
        # and response is wrapped into the following
        # {
        #  "predictions: [
        #    {
        #      "image_bytes": {
        #          "b64": "<b64-encoded>",
        #      },
        #      "gender_and_ages: [
        #        { "label_index": '17', "x": '0.534282', "y": '0.519531', "w":
        #        '0.111255', "h": '0.21875'},
        #      ],
        #      "key": "somekeys",
        #      "type": "insightfacec-detector",
        #    },
        #  ]
        # }

        im1 = base64decode(inputs[0]["image_bytes"]["b64"])
        key = inputs[0]["key"]
        faces = self.app.get(im1)
        gaa = []
        for face in faces:
            gaa.append({
                'gender': int(face['gender']),
                'age': int(face['age']),
                'bbox': face['bbox'].tolist(),
            })
        drawed_im = self.app.draw_on(im1, faces)
        drawed_im_str = base64encode(drawed_im)

        return {
                "predictions": [
                {
                    "image_bytes": {
                        "b64": drawed_im_str,
                    },
                    "gender_and_ages": gaa,
                    "key": key,
                    "type": "face:insightface-detector",
                },
            ]
        }

