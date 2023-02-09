import logging
import uuid
from kubernetes import client, config

CONTAINER_REGISTRY="<Todo_Insert_container_registry_here>"
CONTAINER_TAG="charles-v1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Pytorch Distributed Model Training__")
logger.info("PyTorchJob log information...")

def torch_manifest():
   job_resource = {
      "apiVersion": "kubeflow.org/v1",
      "kind": "PyTorchJob",
      "metadata": {
         "name": "demo-torch-job-" + str(uuid.uuid1()),
         "namespace": "kubeflow"
      },
      "spec": {
         "pytorchReplicaSpecs":{
           "Master": {
              "replicas": 1,
              "restartPolicy": "OnFailure",
              "template": {
                 "metadata": {
                 "annotations": {
                    "sidecar.istio.io/inject": "false"
                 }
                 },
                 "spec": {
                 "containers": [
                    {
                        "name": "pytorch",
                        "image": f"{CONTAINER_REGISTRY}/model_train:{CONTAINER_TAG}",
                        "command": [
                        "python3",
                        "model_train.py",
                        f"--epochs=3",
                        f"--log-interval=128"
                        ],
                     }
                  ],
                  }
                }
            },
            "Worker": {
               "replicas": 2,
               "restartPolicy": "OnFailure",
               "template": {
                  "metadata": {
                  "annotations": {
                     "sidecar.istio.io/inject": "false"
                  }
                  },
                  "spec": {
                  "containers": [
                     {
                        "name": "pytorch",
                        "image": f"{CONTAINER_REGISTRY}/model_train:{CONTAINER_TAG}",
                        "command": [
                        "python3",
                        "model_train.py",
                        f"--epochs=3",
                        f"--log-interval=128"
                        ],
                     }
                  ],
                  }
               }
            }
            }
         }
      }

   return job_resource


def create_torch_job(k8s_api):
  driver_pod_name = "demo-torch-job-" + str(uuid.uuid1())

  k8s_api.create_namespaced_custom_object(
      group="kubeflow.org",
      version="v1",
      namespace="kubeflow",
      plural="pytorchjobs",
      body=job_resource,
  )

  logging.info("PyTorchJob created")
  logging.info(f"PyTorchJob Driver Pod Name: {driver_pod_name}-driver")
  
  
if __name__== "__main__":
   job_resource = torch_manifest()
   logging.info("Creating PyTorchJob CRD manifest...")
   config.load_incluster_config()
   k8s_api = client.CustomObjectsApi()
   logging.info("Creating PyTorchJob resource...")
   step_id = create_torch_job(k8s_api)
