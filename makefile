CURDIR:=$(shell pwd)

PWD = $(shell pwd)
CURDIR:=$(shell pwd)
OS = $(shell go env GOOS)
ARCH = $(shell go env GOARCH)
.PHONY: build clean run

GITCOMMITHASH := $(shell git rev-parse --short HEAD)
GITBRANCHNAME := $(shell git symbolic-ref -q --short HEAD || git describe --tags --exact-match)
GOLDFLAGS += -X handler.__GITCOMMITINFO__=$(GITCOMMITHASH).${GITBRANCHNAME}
GOFLAGS = -ldflags "$(GOLDFLAGS)"

PUBLISHDIR=${CURDIR}/dist
PROJECT_NAME=sunmao


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
	mkdir -pv $(PUBLISHDIR)/lib
	mkdir -pv ${PUBLISHDIR}/conf
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
build-dev: third-dev
	mkdir -pv $(PUBLISHDIR)/lib
	mkdir -pv ${PUBLISHDIR}/conf
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)


prod: build
	mkdir -pv $(PUBLISHDIR)/lib
	
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
pre: build
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
local: build-dev
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	go build -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
dev: third-dev
	mkdir -pv $(PUBLISHDIR)/lib
	cp -aRf third/lib/$(OS)/$(ARCH)/* $(PUBLISHDIR)/lib/
	export GOTRACEBACK=crash  && go build -gcflags=all="-N -l" -o $(PUBLISHDIR)/$(PROJECT_NAME) $(GOFLAGS)
	cp -aRf conf/$@/* ${PUBLISHDIR}/conf
clean:
	rm -rf ./build
	rm -rf ./build_cpp
	rm -rf ./dist
	mkdir -pv build/conf
run: build-dev
	./${PUBLISHDIR}/$(PROJECT_NAME) -config="./build/conf"