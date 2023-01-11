## **Hyperparameter Tuning with katib**  
  
### **Requirements**  
1. make  
2. kubectl
  
### **Implementation**  
Compile the katib pipeline:  
  
```  
make compile-pipeline  
```  
  
Upload the resulting `yaml` file to the kubeflow-ui and create a pipeline run.  
  
  
View the deployed experiments:  
  
```  
make view-experiment  
```  
  
View the deployed pods running the training job:  
  
```  
make view-pods  
```  
  
You can also view the logs of the pods, for the model training job.  
  
```  
microk8s kubectl -n katib-demo logs <insert-the-name-of-the-pod>  
```  
  
The performance of the experiment can be viewed from the katib ui.