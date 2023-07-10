import kfp.dsl as dsl
import kfp
from kfp import components

kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/1.8.5/components/kserve/component.yaml')

@dsl.pipeline(
  name='Insightface KFServing pipeline',
  description=''
)
def kfservingPipeline(
    action='apply',
    model_name='',
    model_uri='',
    namespace='kubeflow-user-example-com',
    runtime_version='v0.0.1',
    runtime_model_name='buffalo_l',
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
            size="10Gi",
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
            size="10Gi",
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
        minReplicas: 0
        maxReplicas: 1
        containers:
        - image: footprintai/insightface:{3}
          name: kfserving-container
          args: ["kserve-main.py"]
          resources:
            limits:
              memory: "8Gi"
              cpu: "1"
              nvidia.com/gpu: 1
            requests:
              memory: "8Gi"
              cpu: "1"
              nvidia.com/gpu: 1
          env:
            - name: STORAGE_URI
              value: {2}
            - name: DEFAULT_MODEL_NAME
              value: {0}
            - name: DEFAULT_RUNTIME_MODEL_NAME
              value: {4}
            - name: CUDA_VISIBLE_DEVICE
              value: "0"
    '''.format(model_name, namespace, model_pvc_uri, runtime_version, runtime_model_name)
        ks = kserve_op(
            action=action,
            inferenceservice_yaml=isvc_yaml
        ).after(initializer)
        ks.set_cpu_request("100m").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
