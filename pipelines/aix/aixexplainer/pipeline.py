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
    action_str = '''{}'''.format(action)
    with dsl.Condition(action == 'delete'):
        model_pvc_name = 'modelpvc-{}'.format(model_name)

        kserveop = kserve_op(
            action=action,
            model_name=model_name,
            namespace=namespace,
        )

        vop = dsl.VolumeOp(
            name=model_pvc_name,
            resource_name=model_pvc_name,
            size="5Gi",
            modes=dsl.VOLUME_MODE_RWM,
            generate_unique_name=False,
            action='apply', # use DELETE will trigger Exception, so we use apply here to "cheat" the volumnop.
        )
        vop.delete()

    with dsl.Condition(action == 'apply'):
        model_pvc_name = 'modelpvc-{}'.format(model_name)

        vop = dsl.VolumeOp(
            name=model_pvc_name,
            resource_name=model_pvc_name,
            size="5Gi",
            modes=dsl.VOLUME_MODE_RWM,
            generate_unique_name=False,
            action=action_str,
            set_owner_reference=False, # by setting False, pvc won't be reclaimed after the workflow has been deleted.
        )

        downloadcli = '''wget '{0}' -O - | tar -xzvf - -C {1}'''.format(model_uri, '/mnt/models')
        initializer = dsl.ContainerOp(
            name="initializer",
            image="library/bash:4.4.23",
            command=["sh", "-c"],
            arguments=[downloadcli],
            pvolumes={"/mnt/models": vop.volume}
        )
        initializer.set_cpu_request("1").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
        model_pvc_uri = '''pvc://{0}/'''.format(model_pvc_name)

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
    tensorflow:
      storageUri: {2}
      image: tensorflow/serving:2.8.0
      resources:
        limits:
          cpu: "1"
          memory: "8Gi"
          nvidia.com/gpu: "1"
        requests:
          cpu: "1"
          memory: "8Gi"
          nvidia.com/gpu: "1"
      env:
      - name: TF_FORCE_GPU_ALLOW_GROWTH
        value: "true"
  explainer:
    containers:
    - name: explainer
      image: footprintai/aix-aixexplainer:{3}
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
          memory: "2Gi"
        requests:
          cpu: "1"
          memory: "2Gi"
'''.format(model_name, namespace, model_pvc_uri, runtime_version)
        ks = kserve_op(
            action=action,
            inferenceservice_yaml=isvc_yaml
        )
        ks.set_cpu_request("100m").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
