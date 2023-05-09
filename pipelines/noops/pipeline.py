import kfp.dsl as dsl
import kfp
from kfp import components

kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/1.8.5/components/kserve/component.yaml')

@dsl.pipeline(
  name='NoOps KFServing pipeline',
  description=''
)
def kfservingPipeline(
    action='apply',
    model_name='noops',
    namespace='kubeflow-user-example-com',
    runtime_version='latest',
):

    isvc_yaml = '''
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: {0}
  namespace: {1}
  annotations:
  serving.kserve.io/deploymentMode: RawDeployment
  serving.kserve.io/autoscalerClass: hpa
  serving.kserve.io/metric: cpu
  serving.kserve.io/targetUtilizationPercentage: "80"
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 1
    containers:
    - name: kfserving-container
      image: docker.io/footprintai/kserve-noops:{2}
      env:
      - name: DEFAULT_MODEL_NAME
        value: {0}
      resources:
        limits:
          memory: "100Mi"
          cpu: "100m"
        requests:
          memory: "100Mi"
          cpu: "100m"

'''.format(model_name, namespace, runtime_version)
    ks = kserve_op(
        action=action,
        inferenceservice_yaml=isvc_yaml
    )
    ks.set_cpu_request("1").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
