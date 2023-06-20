import kfp.dsl as dsl
import kfp
from kfp import components

kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/1.8.5/components/kserve/component.yaml')

@dsl.pipeline(
  name='KFServing pipeline',
  description='A pipeline for KFServing.'
)
def kfservingPipeline(
    action='apply',
    model_name='tensorflow-sample',
    model_uri='gs://kfserving-examples/models/tensorflow/flowers',
    namespace='kubeflow-user-example-com'):

    # flowers model was trained with 5 category: [daisy(雛菊)  dandelion(浦公英)  roses(玫瑰)  sunflowers(向日葵)  tulips(鬱金香)]
    # see details here:  https://github.com/tensorflow/hub/blob/master/examples/colab/image_feature_vector.ipynb
    
    isvc_yaml = '''
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: {0}
  namespace: {1}
  annotations:
  serving.kserve.io/autoscalerClass: hpa
  serving.kserve.io/metric: cpu
  serving.kserve.io/targetUtilizationPercentage: "80"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    tensorflow:
      storageUri: {2}
      image: tensorflow/serving:2.8.0
      resources:
        limits:
          memory: "4Gi"
          cpu: "1"
          nvidia.com/gpu: 1
        requests:
          memory: "4Gi"
          cpu: "1"
          nvidia.com/gpu: 1
'''.format(model_name, namespace, model_uri)
    ks = kserve_op(
        action=action,
        inferenceservice_yaml=isvc_yaml
    )
    ks.set_cpu_request("1").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
