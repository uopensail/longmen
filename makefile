CURDIR:=$(shell pwd)
PWD = $(shell pwd)
OS = $(shell go env GOOS)
ARCH = $(shell go env GOARCH)
.PHONY: build clean run

GITCOMMITHASH := $(shell git rev-parse --short HEAD)
GITBRANCHNAME := $(shell git symbolic-ref -q --short HEAD || git describe --tags --exact-match)
GOLDFLAGS += -X handler.__GITCOMMITINFO__=$(GITCOMMITHASH).${GITBRANCHNAME}
GOFLAGS = -ldflags "$(GOLDFLAGS)"

PUBLISHDIR=${CURDIR}/dist
PROJECT_NAME=longmen
CMAKE_DEFINE := -DPYBIND=OFF
ifeq ($(strip $(TORCH_INSTALL_DIR)),)
    TORCH_CMAKE_DIR :=$(shell python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
else
	TORCH_CMAKE_DIR :=$(TORCH_INSTALL_DIR)/share/cmake
endif

CMAKE_DEFINE += -DCMAKE_PREFIX_PATH=${TORCH_CMAKE_DIR}
TORCH_LIB_DIR=$(TORCH_CMAKE_DIR)/../../lib

all: build-prod
third-dev:
	git submodule update --init --recursive 
	cmake --version
	mkdir -pv build_cpp
	cd build_cpp && cmake ../third/longmen/ -DCMAKE_BUILD_TYPE=Debug ${CMAKE_DEFINE} && make
	mkdir -pv build/
	mkdir -pv third/lib/$(OS)/$(ARCH)/
	find build_cpp -type f -name 'lib*' -exec cp {}  build \;
	find build_cpp -type f -name 'lib*' -exec cp {}  third/lib/$(OS)/$(ARCH)/ \;
third-prod:
	git submodule update --init --recursive 
	cmake --version
	mkdir -pv build_cpp
	cd build_cpp && cmake ../third/longmen/ -DCMAKE_BUILD_TYPE=Release ${CMAKE_DEFINE} && make
	mkdir -pv build/
	mkdir -pv third/lib/$(OS)/$(ARCH)/
	find build_cpp -type f -name 'lib*' -exec cp {}  build \;
	find build_cpp -type f -name 'lib*' -exec cp {}  third/lib/$(OS)/$(ARCH)/ \;

build: third-prod
	mkdir -pv $(PUBLISHDIR)/lib
	mkdir -pv $(PUBLISHDIR)/conf
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	export LD_LIBRARY_PATH=$(TORCH_LIB_DIR) && CGO_LDFLAGS="-L$(TORCH_LIB_DIR)" go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
build-dev: third-dev
	mkdir -pv $(PUBLISHDIR)/lib
	mkdir -pv $(PUBLISHDIR)/conf
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
  
	export GOTRACEBACK=crash && export LD_LIBRARY_PATH=$(TORCH_LIB_DIR) && CGO_LDFLAGS="-L$(TORCH_LIB_DIR)"  go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)


prod: build
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
pre: build
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
local: build-dev
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
dev: third-dev
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
clean:
	rm -rf ./build
	rm -rf ./build_cpp
	rm -rf ./dist
	mkdir -pv build/conf
run: build-dev
	./${PUBLISHDIR}/$(PROJECT_NAME) -config="./build/conf"