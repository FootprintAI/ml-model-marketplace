import kfp.dsl as dsl
import kfp
from kfp import components

kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/1.8.5/components/kserve/component.yaml')

@dsl.pipeline(
  name='Openai Whisper KFServing pipeline',
  description=''
)
def kfservingPipeline(
    action='apply',
    model_name='whisper01',
    model_uri='',
    namespace='kubeflow-user-example-com',
    runtime_version='v0.0.1',
):
    action_str = '''{}'''.format(action)
    model_pvc_name = 'modelpvc-{}'.format(model_name)
    with dsl.Condition(action == 'delete'):
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
        vop = dsl.VolumeOp(
            name=model_pvc_name,
            resource_name=model_pvc_name,
            size="10Gi",
            modes=dsl.VOLUME_MODE_RWM,
            generate_unique_name=False,
            action=action_str,
            set_owner_reference=False, # by setting False, pvc won't be reclaimed after the workflow has been deleted.
        )

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
        - image: footprintai/kserve-whisper:{3}
          name: kfserving-container
          args: ["kserve_main.py"]
          resources:
            limits:
              memory: "1Gi"
              cpu: "100m"
            requests:
              memory: "1Gi"
              cpu: "100m"
          env:
            - name: DEFAULT_MODEL_NAME
              value: {0}
          ports:
          - containerPort: 8080
        - image: onerahmet/openai-whisper-asr-webservice:latest-gpu
          name: whisper-service
          resources:
            limits:
              memory: "12Gi"
              cpu: "4"
              nvidia.com/gpu: 1
            requests:
              memory: "12Gi"
              cpu: "4"
              nvidia.com/gpu: 1
          env:
            - name: ASR_ENGINE
              value: openai_whisper
            - name: ASR_MODEL
              value: large
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: ASR_MODEL_PATH
              value: /mnt/models
            - name: STORAGE_URI
              value: {2}
    '''.format(model_name, namespace, model_pvc_uri, runtime_version)
        ks = kserve_op(
            action=action,
            inferenceservice_yaml=isvc_yaml
        )
        ks.set_cpu_request("1").set_cpu_limit("1").set_memory_request("1G").set_memory_limit("1G")
