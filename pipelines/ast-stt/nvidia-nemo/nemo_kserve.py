import os
import kserve
from typing import Dict
import base64

from opencc import OpenCC
from filepath import modelpath_join

lang2Model = {
    # find pretrained checkpoint at here
    # https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html
    "en": "stt_en_conformer_transducer_large.nemo", 
    # https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_large
    "zh": "stt_zh_conformer_transducer_large.nemo",
    # https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_conformer_transducer_large
}

class STTModel(kserve.Model):
    """ STTModel use a prebuild model to transcript en/mardrin language
    """
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name

    def load(self):
        self.ready = True
        return self.ready

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {
        #   "instances": [
        #     {
        #       "audio_bytes": {
        #           "b64": "<b64-encoded>",
        #       },
        #       "lang": "en", (or "zh")
        #       "key": "somekeys",
        #     },
        #   ],
        # }
        # and response is wrapped into the following
        # {
        #  "predictions: [
        #    {
        #      "audio_bytes": {
        #          "text": "",
        #      },
        #      "key": "somekeys",
        #      "type": "nemo-stt",
        #    },
        #  ]
        # }
        b64data = inputs[0]["audio_bytes"]["b64"]
        key = inputs[0]["key"]
        lang = inputs[0]["lang"]
        if type(lang) != list:
            lang = [lang]
        lang = list(set(lang))
        ts = self._query_ts()
        ts_str = "{}".format(ts)
        query_dir = os.path.join("/tmp", ts_str)
        result_dir = os.path.join("/tmp/result", ts_str)
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        query_file_path = os.path.join(query_dir, "input.wav")
        self._save_audio_to_file(query_file_path, b64data)

        audio_text_dict = {}
        for ln in lang:
            query_result_path = os.path.join(result_dir, ln)
            self._run_stt(ln, query_dir, query_result_path)
            pred_text = self._read_result_text(query_result_path)
            audio_text_dict[ln] = pred_text

        if 'zh' in audio_text_dict:
            cc = OpenCC('s2t') # simplified to traditional
            audio_text_dict['zh'] = cc.convert(audio_text_dict['zh'])

        return {
                "predictions": [
                {
                    "audio_text": audio_text_dict,
                    "key": key,
                    "type": "nemo-stt",
                },
            ]
        }

    def _query_ts(self) -> int:
        import time

        return int(time.time())

    def _save_audio_to_file(self, dstfile:str, b64str: str):
        raw_img_data = base64.b64decode(b64str)
        with open(dstfile, mode='bx') as f:
            f.write(raw_img_data)

    def _run_stt(self, lang: str, query_file:str, result_file:str):
        import subprocess
        multilinecmd = """
python ./examples/asr/transcribe_speech.py \
    model_path={0} \
    audio_dir={1} \
    output_filename={2} \
    batch_size=32 \
    compute_langs=False \
    amp=True \
    append_pred=False
""".format(modelpath_join(lang2Model[lang]), query_file, result_file)
        subprocess.run(multilinecmd, shell=True, check=True,capture_output=True)

    def _read_result_text(self, result_file:str):
        import json

        # result should be with the following structure
        # {"audio_filepath": "/workspace/nemo/datainputs/male.wav", "pred_text": ""}
        with open(result_file) as f:
            result_json = json.load(f)
            return result_json["pred_text"]


