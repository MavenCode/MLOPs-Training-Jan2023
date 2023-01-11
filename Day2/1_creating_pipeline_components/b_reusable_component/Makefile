ENV :=dev
SHELL := /bin/bash

CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq ($(ENV), dev)
	SKAFFOLD_DEFAULT_REPO := "public.ecr.aws/XXXXX"
endif


.PHONY: docker-login
docker-login:
	aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(SKAFFOLD_DEFAULT_REPO)

.PHONY: skaffold-build-local
skaffold-build-local:
	skaffold build  --platform linux/amd64 --default-repo=$(SKAFFOLD_DEFAULT_REPO)

.PHONY: skaffold-build
skaffold-build:
	skaffold build  --platform linux/amd64 --default-repo=$(SKAFFOLD_DEFAULT_REPO) --push --tag charles-latest