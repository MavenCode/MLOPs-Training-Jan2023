# Microk8s Setup

The `Makefile` contains the steps for installing and running Microk8s on the provided Ubuntu boxes for this training

To install Canonical Microk8s on your Ubuntu Training Workstation (Macbook or Windows Laptop), you can follow these steps:
(Note your instance should already have Microk8s bootstrapped. Skip to step 4)

### 1. Install snapd package
This package is required for installing and managing snap packages, which are self-contained applications that can be easily installed and managed on a variety of systems.

### 2. Enable the Microk8s snap channel.
You can do this by running the following command:
```
snap install microk8s --channel=1.22/stable --classic
```

### 3. Install Microk8s
Run the following command to install microk8s on your system
```
snap install microk8s --classic
```

### 4. Start microk8s 
Once Microk8s is installed, you can start it by running the following command:
```
microk8s start
```

### 5. Verify Microk8s is running correctly 
To verify that Microk8s is running correctly, you can use the kubectl command-line tool to check the status of the cluster:
```
microk8s kubectl get all --all-namespaces
```