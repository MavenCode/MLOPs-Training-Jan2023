# Skaffold Setup

## Installations
Install all the required packages by running the commands below.
(Note install skaffold, run the Makefile with "make", and skip remaining steps)

### install Skaffold
```
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 && sudo install skaffold /usr/local/bin/
```
### install make
```
sudo apt install make
```
### install juju
```
sudo apt install juju
```
### install jq
```
sudo apt install jq
```

## Setting Up your local dev environment
All the commands to properly setup the local development environment have been listed on the `Makefile`.

After installing the necessary packages and ensuring that Microk8s is properly installed and running, go through the following steps to set up the local development environment:

### setup kubeconfig
run the command below to setup the kubeconfig
```
make set-cluster-credentials
```

### setup docker credentials
log into the docker container registry by running the command below
```
make docker-login
```

### Skaffold build
run the following command to build a container with Skaffold
```
make skaffold-build-local
```

### Skaffold build and push
run the following command to build and push built containers to the local registry
```
make skaffold-build
```
