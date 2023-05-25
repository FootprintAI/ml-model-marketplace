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
    model_name='aixexplainer',
    model_uri='',
    namespace='kubeflow-user-example-com',
    runtime_version='v0.0.1',
):

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
        containers:
        - name: predictor
          image: aipipeline/rf-predictor:0.4.1
          command: ["python", "-m", "rfserver", "--model_name", {0}]
          imagePullPolicy: Always
      explainer:
        containers:
        - name: explainer
          image: docker.io/footprintai/aix-aixexplainer:{2}
          args:
          - --model_name
          - {0}
          - --explainer_type
          - LimeImages
          - --num_samples
          - "100"
          - --top_labels
          - "10"
          - --min_weight
          - "0.01"
          imagePullPolicy: Always
          resources:
            limits:
              cpu: "1"
              memory: 2Gi
            requests:
              cpu: "1"
              memory: 2Gi
    '''.format(model_name, namespace, runtime_version)
    ks = kserve_op(
            action=action,
            inferenceservice_yaml=isvc_yaml
        )
    ks.set_cpu_request("1").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
