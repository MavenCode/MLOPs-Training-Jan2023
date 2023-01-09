ENV :=dev
SHELL := /bin/zsh #/bin/bash on linux
CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


.PHONY: install-snapd-mac
install-snapd-mac:
	#Install Homebrew
	/bin/zsh -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
	brew install snap

.PHONY: install-microk8s
install-microk8s:
	# Enable the Microk8s snap channel
	snap install microk8s --channel=1.22/stable --classic

	# Install Microk8s
	snap install microk8s --classic

.PHONY: inspect-microk8s
inspect-microk8s:
	microk8s inspect

.PHONY: set-cluster-credentials
set-cluster-credentials:
	sudo usermod -a -G microk8s $(NEWUSER)
	newgrp microk8s
	sudo chown -f -R $(NEWUSER) ~/.kube
	microk8s config | cat - > $(HOME)/.kube/config

.PHONY: microk8s-addons
microk8s-addons:
	microk8s enable dns storage ingress metallb:10.64.140.43-10.64.140.49
	microk8s status --wait-ready

.PHONY: start-microk8s
start-microk8s:
	microk8s start

.PHONY: stop-microk8s
stop-microk8s:
	microk8s stop

.PHONY: verify-microk8s
verify-microk8s:
	microk8s kubectl get all --all-namespaces

