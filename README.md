# ml-model-marketplace

In this project, we wrapped public accessible ml/dl models with kubeflow pipeline.

#### Pipeline ####

each pipeline should under a named folder with a file named `pipeline.py`, this is the only reference the kfp.compiler can find you and build a `workflow` resource dynamically.

#### How to build #####

Simple launch the following command to build all manifests that we can find with the following pattern `find pipelines -name "*pipeline.py"`

```
make gen-manifests

```
