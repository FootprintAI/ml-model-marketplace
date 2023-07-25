import cv2
import numpy as np
import kserve
from paddleocr import PaddleOCR, draw_ocr
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

class PaddleOCRModel(kserve.Model):
    def __init__(self, name: str, lang: str):
        super().__init__(name)
        self.name = name
        self.lang = lang
        self.app = None

    def load(self):
        self.app = PaddleOCR(use_angle_cls=True, lang=self.lang,
                             det_model_dir='/mnt/models/det',
                             rec_model_dir='/mnt/models/rec',
                             cls_model_dir='/mnt/models/cls') 
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
        #      "en": "",
        #      "key": "somekeys",
        #      "type": "paddle-oct-detector",
        #    },
        #  ]
        # }

        im1 = base64decode(inputs[0]["image_bytes"]["b64"])
        key = inputs[0]["key"]
        result = self.app.ocr(im1, cls=True)
        # [[[[[371.0, 306.0], [673.0, 310.0], [671.0, 455.0], [369.0, 451.0]], ('SEE', 0.5028034448623657)], [[[419.0, 560.0], [615.0, 560.0], [615.0, 597.0], [419.0, 597.0]], ('HING KEE', 0.980820894241333)], [[[379.0, 621.0], [654.0, 621.0], [654.0, 653.0], [379.0, 653.0]], ('RESTAURANT', 0.9898968935012817)]]]
        result = result[0]

        if not result:
            return {
                "predictions": [
                {
                    "image_bytes": {
                        "b64": inputs[0]["image_bytes"]["b64"],
                    },
                    "text_and_positions": [],
                    "key": key,
                    "type": "ocr:paddleocr",
                }, ]
            }

        boxes = [line[0] for line in result]
        # [[[371.0, 306.0], [673.0, 310.0], [671.0, 455.0], [369.0, 451.0]], [[419.0, 560.0], [615.0, 560.0], [615.0, 597.0], [419.0, 597.0]], [[379.0, 621.0], [654.0, 621.0], [654.0, 653.0], [379.0, 653.0]]]
        txts = [line[1][0] for line in result]
        # ['SEE', 'HING KEE', 'RESTAURANT']
        scores = [line[1][1] for line in result]
        # [0.5028034448623657, 0.980820894241333, 0.9898968935012817]
        text_and_position = []
        for i in range(len(boxes)):
            text_and_position.append({
                'text': txts[i],
                'position': boxes[i],
                'score': scores[i],
            })
        drawed_im = draw_ocr(im1, boxes, txts, scores, font_path='./fonts/simfang.ttf')
        drawed_im_str = base64encode(drawed_im)

        return {
                "predictions": [
                {
                    "image_bytes": {
                        "b64": drawed_im_str,
                    },
                    "text_and_positions": text_and_position,
                    "key": key,
                    "type": "ocr:paddleocr",
                },
            ]
        }
if __name__ == '__main__' :
    app = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='/mnt/models') 
    img_path = 'testocr.png'
    im1 = imread(img_path)
    result = app.ocr(im1, cls=True)
    #for idx in range(len(result)):
    #    res = result[idx]
    #    for line in res:
    #        print(line)
    #        print("\n")
    # result would look like these:
    #[[[38.0, 91.0], [578.0, 92.0], [578.0, 116.0], [38.0, 115.0]], ('This is a lot of 12 point text to test the', 0.9896008372306824)]
    #[[[35.0, 124.0], [617.0, 127.0], [617.0, 153.0], [35.0, 150.0]], ('ocr code and see if it works on all types', 0.9963929653167725)]
    #[[[35.0, 159.0], [224.0, 159.0], [224.0, 186.0], [35.0, 186.0]], ('of file format.', 0.9996032118797302)]
    #[[[36.0, 193.0], [584.0, 194.0], [584.0, 222.0], [36.0, 221.0]], ('The guick brown dog jumped over the', 0.9678623676300049)]
    #[[[34.0, 228.0], [586.0, 228.0], [586.0, 258.0], [34.0, 258.0]], ('Iazy fox. The quick brown dog jumped', 0.9852568507194519)]
    #[[[35.0, 262.0], [596.0, 263.0], [596.0, 291.0], [35.0, 290.0]], ('over the lazy fox. The quick brown dog', 0.9771052002906799)]
    #[[[43.0, 297.0], [562.0, 296.0], [562.0, 323.0], [43.0, 324.0]], ('jumped over the lazy fox. The quick', 0.9919435977935791)]
    #[[[36.0, 331.0], [561.0, 331.0], [561.0, 358.0], [36.0, 358.0]], ('brown dog jumped over the lazy fox.', 0.991053581237793)]
    boxes = [line[0] for res in result for line in res]
    txts = [line[1][0] for res in result for line in res]
    scores = [line[1][1] for res in result for line in res]

    text_and_position = []
    for i in range(len(boxes)):
        text_and_position.append({
            'text': txts[i],
            'position': boxes[i],
            'score': scores[i],
        })
    # print(text_and_position)
    # [{'text': 'This is a lot of 12 point text to test the', 'position': [[38.0, 91.0], [578.0, 92.0], [578.0, 116.0], [38.0, 115.0]], 'score': 0.9896008372306824}, {'text': 'ocr code and see if it works on all types', 'position': [[35.0, 124.0], [617.0, 127.0], [617.0, 153.0], [35.0, 150.0]], 'score': 0.9963929653167725}, {'text': 'of file format.', 'position': [[35.0, 159.0], [224.0, 159.0], [224.0, 186.0], [35.0, 186.0]], 'score': 0.9996032118797302}, {'text': 'The guick brown dog jumped over the', 'position': [[36.0, 193.0], [584.0, 194.0], [584.0, 222.0], [36.0, 221.0]], 'score': 0.9678623676300049}, {'text': 'Iazy fox. The quick brown dog jumped', 'position': [[34.0, 228.0], [586.0, 228.0], [586.0, 258.0], [34.0, 258.0]], 'score': 0.9852568507194519}, {'text': 'over the lazy fox. The quick brown dog', 'position': [[35.0, 262.0], [596.0, 263.0], [596.0, 291.0], [35.0, 290.0]], 'score': 0.9771052002906799}, {'text': 'jumped over the lazy fox. The quick', 'position': [[43.0, 297.0], [562.0, 296.0], [562.0, 323.0], [43.0, 324.0]], 'score': 0.9919435977935791}, {'text': 'brown dog jumped over the lazy fox.', 'position': [[36.0, 331.0], [561.0, 331.0], [561.0, 358.0], [36.0, 358.0]], 'score': 0.991053581237793}]
    drawed_im = draw_ocr(im1, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    drawed_im_str = base64encode(drawed_im)
    
    
