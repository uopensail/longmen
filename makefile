# ==============================================================================
# Makefile for LongMen Service
# ==============================================================================

# ==============================================================================
# Variables
# ==============================================================================

# Project information
PROJECT_NAME := longmen
VERSION := 1.0.0

# Directories
CURDIR := $(shell pwd)
DISTDIR := $(CURDIR)/dist
CONFDIR := $(CURDIR)/conf

# Go environment
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
GO := go

# Git information for version tracking
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH := $(shell git symbolic-ref -q --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')

# Linker flags for embedding version information
LDFLAGS := -s -w \
	-X 'main.AppVersion=$(VERSION)' \
	-X 'main.GitCommit=$(GIT_COMMIT)' \
	-X 'main.GitBranch=$(GIT_BRANCH)' \
	-X 'main.BuildTime=$(BUILD_TIME)'

# Binary output
BINARY := $(DISTDIR)/$(PROJECT_NAME)

# ==============================================================================
# Phony Targets
# ==============================================================================

.PHONY: all build clean run test fmt vet check help \
        dev prod local pre deps tidy version prepare

# Default target
.DEFAULT_GOAL := help

# ==============================================================================
# Main Targets
# ==============================================================================

## all: Build the project (default)
all: build

## build: Build the binary
build: clean prepare
	@echo "Building $(PROJECT_NAME)..."
	@$(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY)
	@echo "Build complete: $(BINARY)"

## clean: Remove build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf $(DISTDIR)
	@rm -f coverage.out
	@echo "Clean complete"

## run: Build and run the service
run: build
	@echo "Running $(PROJECT_NAME)..."
	@$(BINARY) -config=$(DISTDIR)/conf/config.toml

# ==============================================================================
# Environment Targets
# ==============================================================================

## dev: Build for development environment
dev: clean prepare
	@echo "Building for development..."
	@$(GO) build -gcflags="all=-N -l" -ldflags "$(LDFLAGS)" -o $(BINARY)
	@cp -rf $(CONFDIR)/dev/* $(DISTDIR)/conf/
	@echo "Development build complete"

## local: Build for local testing
local: clean prepare
	@echo "Building for local environment..."
	@$(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY)
	@cp -rf $(CONFDIR)/local/* $(DISTDIR)/conf/
	@echo "Local build complete"

## pre: Build for pre-production environment
pre: clean prepare
	@echo "Building for pre-production..."
	@$(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY)
	@cp -rf $(CONFDIR)/pre/* $(DISTDIR)/conf/
	@echo "Pre-production build complete"

## prod: Build for production environment
prod: clean prepare
	@echo "Building for production..."
	@$(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY)
	@cp -rf $(CONFDIR)/prod/* $(DISTDIR)/conf/
	@echo "Production build complete"

# ==============================================================================
# Code Quality Targets
# ==============================================================================

## fmt: Format Go code
fmt:
	@echo "Formatting code..."
	@$(GO) fmt ./...
	@echo "Format complete"

## vet: Run go vet
vet:
	@echo "Running go vet..."
	@$(GO) vet ./...
	@echo "Vet complete"

## check: Run all checks (fmt, vet, test)
check: fmt vet test
	@echo "All checks passed"

# ==============================================================================
# Dependency Management
# ==============================================================================

## deps: Download dependencies
deps:
	@echo "Downloading dependencies..."
	@$(GO) mod download
	@echo "Dependencies downloaded"

## tidy: Tidy dependencies
tidy:
	@echo "Tidying dependencies..."
	@$(GO) mod tidy
	@echo "Dependencies tidied"

# ==============================================================================
# Utility Targets
# ==============================================================================

## version: Show version information
version:
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Git Commit: $(GIT_COMMIT)"
	@echo "Git Branch: $(GIT_BRANCH)"
	@echo "Build Time: $(BUILD_TIME)"
	@echo "Go Version: $(shell $(GO) version)"
	@echo "OS/Arch: $(GOOS)/$(GOARCH)"

## prepare: Prepare build directories
prepare:
	@mkdir -p $(DISTDIR)/conf

## help: Show this help message
help:
	@echo "$(PROJECT_NAME) Makefile"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build the binary"
	@echo "  clean          Remove build artifacts"
	@echo "  run            Build and run the service"
	@echo ""
	@echo "  dev            Build for development"
	@echo "  local          Build for local testing"
	@echo "  pre            Build for pre-production"
	@echo "  prod           Build for production"
	@echo ""
	@echo "  test           Run tests"
	@echo "  test-coverage  Run tests with coverage"
	@echo ""
	@echo "  fmt            Format code"
	@echo "  vet            Run go vet"
	@echo "  check          Run all checks"
	@echo ""
	@echo "  deps           Download dependencies"
	@echo "  tidy           Tidy dependencies"
	@echo ""
	@echo "  version        Show version information"
	@echo "  help           Show this help message"
	@echo ""
