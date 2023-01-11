ENV :=dev
SHELL := /bin/bash

CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

compile-pipeline:
	python3 pipeline/katib_pipeline.py

view-experiment:
	microk8s kubectl -n admin get experiments

view-pods:
	microk8s kubectl -n admin get pods