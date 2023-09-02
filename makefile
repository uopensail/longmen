PWD = $(shell pwd)
OS = $(shell go env GOOS)
ARCH = $(shell go env GOARCH)
PROJECT_NAME = longmen
GITHASH := $(shell git rev-parse --short HEAD)
BRANCH_NAME := $(shell git symbolic-ref -q --short HEAD || git describe --tags --exact-match)
GOLDFLAGS += -X app.__GITHASH__=$(GITHASH).$(BRANCH_NAME)
GOFLAGS = -ldflags "$(GOLDFLAGS)"
.PHONY: build clean run

PUBLISH_DIR=build

all: build-prod
third-dev:
	cmake --version
	mkdir -pv build_cpp
	cd build_cpp && cmake ../third/longmen/ -DCMAKE_BUILD_TYPE=Debug && make
	mkdir -pv build/
	cp build_cpp/lib* build/
	mkdir -pv third/lib/$(OS)/$(ARCH)/
	cp build_cpp/lib* third/lib/$(OS)/$(ARCH)/
third-prod:
	cmake --version
	mkdir -pv build_cpp
	cd build_cpp && cmake ../third/longmen/ -DCMAKE_BUILD_TYPE=Release && make
	mkdir -pv build/
	cp build_cpp/lib* build/
	mkdir -pv third/lib/$(OS)/$(ARCH)/
	cp build_cpp/lib* third/lib/$(OS)/$(ARCH)/
build: third-prod
	mkdir -pv $(PUBLISH_DIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISH_DIR)/lib/
	go build -o $(PUBLISH_DIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -rf conf/prod/* $(PUBLISH_DIR)/conf
build-dev: third-dev
	mkdir -pv $(PUBLISH_DIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISH_DIR)/lib/
	export GOTRACEBACK=crash  && go build  -gcflags=all="-N -l" -o $(PUBLISH_DIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -rf conf/dev/* $(PUBLISH_DIR)/conf
build-prod: build
	mkdir -pv $(PUBLISH_DIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISH_DIR)/lib/
	go build -o $(PUBLISH_DIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -rf conf/prod/* $(PUBLISH_DIR)/conf
clean:
	rm -rf ./build
	rm -rf ./build_cpp
	mkdir -pv build/conf
run: build-dev
	./${PUBLISH_DIR}/$(PROJECT_NAME) -config="./build/conf"