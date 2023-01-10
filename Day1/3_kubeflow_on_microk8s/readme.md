# Kubeflow setup

## Installation

Run the following commands to install kubeflow on microk8s:
(Note skip installing juju as it has already been installed)

### install juju

```
sudo snap install juju --classic
```

### connect juju to bootstap

```
juju bootstrap microk8s
```
  
### Add controller Juju Controller
```
juju add-model kubeflow
```

### Deploy Kubeflow

```
cd
juju deploy ./kubeflow.yaml –trust
```

### Check installation status

Wait till all the microservices for kubeflow are installed and running on the microk8s cluster. you can check the status
by running the command
```
watch -c juju status --color
```
## configuration

After kubeflow is installed and all microservices are online, go through the following steps to configure the kubeflow deployment 
and access the `kubeflow Dashboard`

### Get Ingress Gateway IP
```
microk8s kubectl -n kubeflow get svc istio-ingressgateway-workload -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```
The Gateway IP we get for this step is used in the proceeding 2 steps. the IP is set to `10.64.140.43` in the following steps. 
update this value in the proceeding steps if you get a different Gateway IP 

### Configure Dex Auth
```
juju config dex-auth public-url=http://10.64.140.43.nip.io
```

### Configure OIDC

```
juju config oidc-gatekeeper public-url=http://10.64.140.43.nip.io
```

### Set Dex Auth Username
```
juju config dex-auth static-username=admin
```
### Set Dex Auth Password
```
juju config dex-auth static-password=admin
```

## Kubeflow Dashboard

Once kubeflow is installed and configured, you can access the `Kubeflow Dashboard` by navigating to [http://10.64.140.43.nip.io](http://10.64.140.43.nip.io). 
use "admin" as the username and password to log into the Dashboard

