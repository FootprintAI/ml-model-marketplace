import base64
import unittest

from nemo_kserve import STTModel

class TestSTTMethods(unittest.TestCase):

    def test_predict(self):
        sttmodel = STTModel('testmodelname')

        with open('sample.wav', 'rb') as fd:
            audio_bytes = fd.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        request = {
            "instances": [
                {
                    "audio_bytes": {
                        "b64": audio_b64,
                    },
                    "lang": ["en"],
                    "key": "1",
                },
            ]
        }
        response = sttmodel.predict(request)
        print(response)

        #self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()
