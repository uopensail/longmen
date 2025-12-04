# Longmen

A high-performance ONNX-based ranking inference service built with C++ and Go.

## Overview

Longmen provides real-time ranking inference using ONNX models with efficient sparse embedding lookup and graceful resource management.

## Project Structure
```bash
longmen/
├── main.go              # Service entry point
├── api/                 # gRPC API definitions
│   ├── api.proto        # Protocol buffer definitions
│   ├── api.pb.go        # Generated Go code
│   ├── api_grpc.pb.go   # Generated gRPC code
│   └── proto.sh         # Script to generate protobuf code
├── services/            # Service layer - gRPC/HTTP handlers
│   └── services.go
├── ranker/              # Ranking engine - inference logic (Go + CGO)
│   └── ranker.go
├── config/              # Configuration management
│   └── config.go
├── third/               # Third-party C++ libraries
│   ├── CMakeLists.txt   # CMake build configuration
│   ├── longmen/         # C++ inference engine
│   └── minia/           # Minia library
├── conf/                # Configuration files
│   └── local/           # Local environment config
├── dist/                # Distribution directory
│   ├── conf/            # Deployed config
│   ├── logs/            # Log files
│   └── longmen          # Compiled binary
├── test/                # Test scripts
│   └── test.py
└── makefile             # Build automation
```

## Prerequisites

- Go 1.24+
- C++20 compiler
- ONNX Runtime
- CMake 3.16+
- Etcd (for service registration)

## Build


### Build C++ components
```bash
cd third
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
cd ../..
```

### Build Go service
```bash
make prod
```

## Configuration

Create configuration file in `conf/local/config.toml`:

```toml
[server]
project_name = "longmen"
grpc_port = 9527          # gRPC service port
http_port = 9528          # HTTP API port
prome_port = 9529         # Prometheus metrics port
pprof_port = 9530         # pprof profiling port
debug = true              # Enable debug mode

[server.register.etcd]
name = "test"             # Service name for registration
endpoints = ["http://127.0.0.1:2379"]  # Etcd endpoints

[env]
workdir = "/tmp/dnnrec_dump"              # Working directory for temp files
itemdir = "/tmp/items"  # Item data directory
```
### Configuration Parameters

**[server]**
- `project_name`: Service identifier
- `grpc_port`: gRPC service port (default: 9527)
- `http_port`: HTTP API port (default: 9528)
- `prome_port`: Prometheus metrics port (default: 9529)
- `pprof_port`: Go pprof profiling port (default: 9530)
- `debug`: Enable debug logging

**[server.register.etcd]**
- `name`: Service registration name in Etcd
- `endpoints`: Etcd server addresses

**[env]**
- `workdir`: Directory for temporary files
- `itemdir`: Directory containing item data

## Run

# Run compiled binary
```bash
./dist/longmen -config conf/local/config.toml
```

# With custom log directory
```bash
./dist/longmen -config conf/local/config.toml -log ./dist/logs
```

## Verify Service

# Check gRPC service
```bash
grpcurl -plaintext localhost:9527 list
```

# Check HTTP health
```bash
curl http://localhost:9528/health

# Check Prometheus metrics
curl http://localhost:9529/metrics

# Check pprof
curl http://localhost:9530/debug/pprof/

```
## API Usage

### gRPC Example
```go
import (
    "context"
    "time"
    "google.golang.org/grpc"
    pb "path/to/api"
)

conn, err := grpc.Dial("localhost:9527", grpc.WithInsecure())
defer conn.Close()

client := pb.NewRankServiceClient(conn)
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

resp, err := client.Rank(ctx, &pb.RankRequest{
    Features: features,
})
```
### HTTP Example
```bash
curl -X POST http://localhost:9528/api/rank \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```
## Development

### Generate Protobuf Code
```bash
cd api
./proto.sh
```
### Run Tests

```bash
# Python test
python test/test.py
```

### Debug Mode

Set `debug = true` in config file to enable:
- Verbose logging
- Request/response tracing
- Performance profiling

## Deployment

### Prepare Directories

mkdir -p /var/lib/longmen/{dump,items,logs}

### Production Configuration
```toml
[server]
project_name = "longmen"
grpc_port = 9527
http_port = 9528
prome_port = 9529
pprof_port = 9530
debug = false

[server.register.etcd]
name = "longmen-prod"
endpoints = ["http://etcd1:2379", "http://etcd2:2379"]

[env]
workdir = "/var/lib/longmen/dump"
itemdir = "/var/lib/longmen/items"
````

### Start Service
```bash
./dist/longmen -config conf/prod/config.toml -log /var/lib/longmen/logs
```
## Monitoring

# View logs
```bash
tail -f dist/logs/longmen.log

# Check metrics
curl http://localhost:9529/metrics | grep longmen

# Memory profiling
go tool pprof http://localhost:9530/debug/pprof/heap

# CPU profiling
go tool pprof http://localhost:9530/debug/pprof/profile
```

## Graceful Shutdown

The service handles these signals for graceful shutdown:
- SIGINT (Ctrl+C)
- SIGTERM
- SIGQUIT

All resources are properly cleaned up before exit.

## License

GNU Affero General Public License v3.0 (AGPL-3.0)
