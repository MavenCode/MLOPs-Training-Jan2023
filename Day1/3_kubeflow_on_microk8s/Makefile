ENV :=dev
SHELL := /bin/bash
CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))



.PHONY: microk8s-install
microk8s-install:
	sudo snap install microk8s –-classic –-channel=1.22/stable

.PHONY: microk8s-inspect
microk8s-inspect:
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

.PHONY: microk8s-start
microk8s-start:
	microk8s start

.PHONY: microk8s-stop
microk8s-stop:
	microk8s stop

