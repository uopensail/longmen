PWD = $(shell pwd)
OS = $(shell go env GOOS)
ARCH = $(shell go env GOARCH)
PROJECT_NAME = longmen
VERSION := $(shell git rev-parse --short HEAD)
BRANCH_NAME := $(shell git symbolic-ref -q --short HEAD || git describe --tags --exact-match)
GOLDFLAGS += -X http._GITHASH_=$(VERSION).$(BRANCH_NAME)
GOFLAGS = -ldflags "$(GOLDFLAGS)"
.PHONY: build clean run

PUBLISH_DIR=build
all: build-dev

third-build-release:
	cmake --version
	git submodule update --init --recursive
	mkdir -p cpp/cmake-build-release
	mkdir -p lib
	cd cpp/cmake-build-release && cmake .. && make && cp -r liblongmen.* ../../lib

third-build-debug:
	cmake --version
	git submodule update --init --recursive
	mkdir -p cpp/cmake-build-debug
	cd cpp/cmake-build-debug && cmake .. && make && cp -r cpp/cmake-build-debug/liblongmen.* ../../lib

build-go-debug:
	git submodule update --init --recursive
	mkdir -pv $(PUBLISH_DIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISH_DIR)/lib/
	export GOTRACEBACK=crash  && go build  -gcflags=all="-N -l" -o $(PUBLISH_DIR)/$(PROJECT_NAME) $(GOFLAGS)
build-go-release:
	git submodule update --init --recursive
	mkdir -pv $(PUBLISH_DIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISH_DIR)/lib/
	go build -o $(PUBLISH_DIR)/$(PROJECT_NAME) $(GOFLAGS)

build-dev: clean build-go-debug
	cp -rf conf/dev $(PUBLISH_DIR)/conf
	bash gen_version.sh $@ build/conf/auto_gen_version.json
build-test: clean build-go-debug
	cp -rf conf/test $(PUBLISH_DIR)/conf
	bash gen_version.sh $@ build/conf/auto_gen_version.json
build-pre: clean build-go-release
	cp -rf conf/pre $(PUBLISH_DIR)/conf
	bash gen_version.sh $@ build/conf/auto_gen_version.json
build-prod: clean build-go-release
	cp -rf conf/prod $(PUBLISH_DIR)/conf
	bash gen_version.sh $@ build/conf/auto_gen_version.json
build-prod-ali: clean build-go-release
	cp -rf conf/prod-ali $(PUBLISH_DIR)/conf
	bash gen_version.sh $@ build/conf/auto_gen_version.json
clean:
	rm -rf ./build
	rm -rf ./build_tmp
	gcc -v
run: build-dev
	./build/$(PROJECT_NAME) -config="./build/conf"