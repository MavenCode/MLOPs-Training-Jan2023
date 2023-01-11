ENV :=dev
SHELL := /bin/bash

CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq ($(ENV), dev)
	SKAFFOLD_DEFAULT_REPO := "localhost:32000" #TODO: replace with container registry
endif


.PHONY: docker-login
docker-login:
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 000654207548.dkr.ecr.us-west-2.amazonaws.com

.PHONY: skaffold-build-local
skaffold-build-local:
	skaffold build  --platform linux/amd64 --default-repo=$(SKAFFOLD_DEFAULT_REPO)

.PHONY: skaffold-build
skaffold-build:
	skaffold build  --platform linux/amd64 --default-repo=$(SKAFFOLD_DEFAULT_REPO) --push