import kserve
import requests
from typing import Dict

INFERENCE_ENDPOINT="127.0.0.1"
INFERENCE_PORT = "9000"

def base64decode(s:str):
    import base64
    raw_original = base64.b64decode(s)
    return raw_original

def base64encode(raw_origin) -> str:
    import base64
    return base64.b64encode(raw_origin)

class WhisperModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    def load(self):
        # marked as ready
        self.ready = True

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        # convert to zhtw
        from opencc import OpenCC

        cc = OpenCC('s2t')
        inputs["predictions"][0]["text"] = cc.convert(inputs["predictions"][0]["text"])
        for seg in inputs["predictions"][0]["segments"]:
            for word in seg["words"]:
                word["word"] = cc.convert(word["word"])
            seg["text"] = cc.convert(seg["text"])
        return inputs

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {
        #   "instances": [
        #     {
        #       "audio_bytes": {
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
        #      "audio_bytes": {
        #          "b64": "<b64-encoded>",
        #      },
        #      "text": <full-text>,
        #      "segments": [{
        #        "id": int,
        #        "start": sec1,
        #        "end": sec2,
        #        "text": <segment-text>,
        #        "tokens": [],
        #      }],
        #      "key": "somekeys",
        #      "type": "whisper",
        #    },
        #  ]
        # }


        key = inputs[0]["key"]
        audio1 = base64decode(inputs[0]["audio_bytes"]["b64"])

        headers = {
            'accept': 'application/json',
        }
        params = {
            "encode": "true",
            "task": "transcribe",
            "language": "zh",
            "word_timestamps": "true",
            "output": "json",
        }
        files = {
            "audio_file": ("input.mp3", audio1, "audio/mpeg"),
        }

        url = "http://{0}:{1}/asr".format(INFERENCE_ENDPOINT, INFERENCE_PORT)
        response = requests.post(url, params=params, files=files, headers=headers)
        response_json = response.json()
        # output:
        # {
        #   "text": "abcd",
        #   "segments": [{
        #     "id": int,
        #     "start": sec1,
        #     "end": sec2,
        #     "text": "abcd",
        #     "tokens": [],
        #   }]
        # }

        return {
            "predictions": [
                {
                    "audio_bytes": {
                        "b64": inputs[0]["audio_bytes"]["b64"],
                    },
                    "text": response_json["text"],
                    "segments": response_json["segments"],
                    "key": key,
                    "type": "stt",
                },
            ]
        }
