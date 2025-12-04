package services

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/uopensail/longmen/api"
	"github.com/uopensail/longmen/config"
	"github.com/uopensail/longmen/ranker"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	etcdclient "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

// ==============================================================================
// Error Definitions
// ==============================================================================

var (
	// ErrEmptyRequest indicates the request has no entries
	ErrEmptyRequest = errors.New("request entries cannot be empty")

	// ErrInvalidRequest indicates the request format is invalid
	ErrInvalidRequest = errors.New("invalid request format")

	// ErrRankerNotInitialized indicates the ranker is not ready
	ErrRankerNotInitialized = errors.New("ranker not initialized")
)

// ==============================================================================
// Services - Main Service Handler
// ==============================================================================

// Services provides the main service implementation for the ranking API
//
// Implements both gRPC (api.RankServer) and HTTP (Gin) interfaces.
// Thread-safe and can handle concurrent requests.
type Services struct {
	api.UnimplementedRankServer
	ranker *ranker.Ranker // Core ranking engine
}

// NewServices creates a new service instance
//
// Initializes the ranker with configuration from the global config instance.
// Panics if ranker initialization fails.
func NewServices() *Services {
	zlog.LOG.Info("Creating services instance")

	// Create ranker instance
	rnk := ranker.NewRanker(
		config.AppConfigInstance.WorkDir,
		config.AppConfigInstance.ItemDir,
	)

	zlog.LOG.Info("Services instance created successfully",
		zap.String("workdir", config.AppConfigInstance.WorkDir),
		zap.String("itemdir", config.AppConfigInstance.ItemDir))

	return &Services{
		ranker: rnk,
	}
}

// ==============================================================================
// Service Lifecycle Management
// ==============================================================================

// Init initializes the service with etcd and service registration
//
// This method can be used to register the service with etcd for service discovery.
// Currently a placeholder for future implementation.
func (srv *Services) Init(etcdName string, etcdCli *etcdclient.Client, reg utils.Register) {
	zlog.LOG.Info("Initializing service",
		zap.String("etcd_name", etcdName))

	// TODO: Implement service registration logic
	// - Register service with etcd
	// - Setup health check callbacks
	// - Initialize metrics

	zlog.LOG.Info("Service initialized successfully")
}

// Close gracefully shuts down the service
//
// Releases all resources including the ranker instance.
// Should be called during application shutdown.
func (srv *Services) Close() {
	zlog.LOG.Info("Closing services")

	if srv.ranker != nil {
		srv.ranker.Close()
		srv.ranker = nil
	}

	zlog.LOG.Info("Services closed successfully")
}

// ==============================================================================
// gRPC Service Registration
// ==============================================================================

// RegisterGrpc registers the gRPC service handlers
//
// Registers the RankServer implementation with the gRPC server.
func (srv *Services) RegisterGrpc(grpcS *grpc.Server) {
	zlog.LOG.Info("Registering gRPC services")

	api.RegisterRankServer(grpcS, srv)

	zlog.LOG.Info("gRPC services registered successfully")
}

// ==============================================================================
// HTTP Service Registration
// ==============================================================================

// RegisterGinRouter registers HTTP API routes with Gin
//
// Registers the following endpoints:
//   - POST /api/v1/rank - Ranking endpoint
//   - GET  /api/v1/health - Health check endpoint
//   - GET  /api/v1/version - Version information endpoint
func (srv *Services) RegisterGinRouter(ginEngine *gin.Engine) {
	zlog.LOG.Info("Registering HTTP routes")

	apiV1 := ginEngine.Group("/api/v1")
	{
		// Ranking endpoint
		apiV1.POST("/rank", srv.RankHandler)

		// Health check endpoint
		apiV1.GET("/health", srv.HealthHandler)

		// Version endpoint
		apiV1.GET("/version", srv.VersionHandler)
	}

	zlog.LOG.Info("HTTP routes registered successfully")
}

// ==============================================================================
// HTTP Handlers
// ==============================================================================

// RankHandler handles HTTP ranking requests
//
// Endpoint: POST /api/v1/rank
// Request body: JSON-encoded api.Request
// Response: JSON-encoded api.Response or error
func (srv *Services) RankHandler(c *gin.Context) {
	stat := prome.NewStat("Services.RankHandler")
	defer stat.End()

	// Parse request
	request := &api.Request{}
	if err := c.ShouldBindJSON(request); err != nil {
		stat.MarkErr()
		zlog.LOG.Error("Failed to parse request body",
			zap.Error(err),
			zap.String("remote_addr", c.ClientIP()))

		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid request format",
			"message": err.Error(),
		})
		return
	}

	// Log request details
	zlog.LOG.Debug("Received ranking request",
		zap.Int("entry_count", len(request.Entries)),
		zap.Int("features_len", len(request.Features)),
		zap.String("remote_addr", c.ClientIP()))

	// Perform ranking
	resp, err := srv.Rank(c.Request.Context(), request)
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("Ranking failed",
			zap.Error(err),
			zap.Int("entry_count", len(request.Entries)))

		// Determine HTTP status code based on error type
		statusCode := http.StatusInternalServerError
		if errors.Is(err, ErrEmptyRequest) || errors.Is(err, ErrInvalidRequest) {
			statusCode = http.StatusBadRequest
		}

		c.JSON(statusCode, gin.H{
			"error":   "ranking failed",
			"message": err.Error(),
		})
		return
	}

	// Return successful response
	zlog.LOG.Debug("Ranking completed successfully",
		zap.Int("entry_count", len(resp.Entries)),
		zap.Int64("version", resp.Version))

	c.JSON(http.StatusOK, resp)
}

// HealthHandler handles health check requests
//
// Endpoint: GET /api/v1/health
// Response: JSON with service health status
func (srv *Services) HealthHandler(c *gin.Context) {
	if srv.ranker == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":  "unhealthy",
			"message": "ranker not initialized",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
	})
}

// VersionHandler returns version information
//
// Endpoint: GET /api/v1/version
// Response: JSON
func (srv *Services) VersionHandler(c *gin.Context) {
	if srv.ranker == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "ranker not initialized",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{})
}

// ==============================================================================
// gRPC Handlers
// ==============================================================================

// Rank performs ranking inference via gRPC
//
// Implements the api.RankServer.Rank method.
// Thread-safe and can handle concurrent requests.
func (srv *Services) Rank(ctx context.Context, req *api.Request) (*api.Response, error) {
	stat := prome.NewStat("Services.Rank")
	defer stat.End()

	// Validate ranker is initialized
	if srv.ranker == nil {
		stat.MarkErr()
		zlog.LOG.Error("Ranker not initialized")
		return nil, status.Error(codes.Unavailable, ErrRankerNotInitialized.Error())
	}

	// Validate request
	if req == nil {
		stat.MarkErr()
		zlog.LOG.Error("Received nil request")
		return nil, status.Error(codes.InvalidArgument, ErrInvalidRequest.Error())
	}

	if len(req.Entries) == 0 {
		stat.MarkErr()
		zlog.LOG.Warn("Received empty request")
		return nil, status.Error(codes.InvalidArgument, ErrEmptyRequest.Error())
	}

	if len(req.Features) == 0 {
		stat.MarkErr()
		zlog.LOG.Warn("Received request with empty features")
		return nil, status.Error(codes.InvalidArgument, "user features cannot be empty")
	}

	// Set metrics
	stat.SetCounter(len(req.Entries))

	// Log request
	zlog.LOG.Debug("Processing gRPC ranking request",
		zap.Int("entry_count", len(req.Entries)),
		zap.Int("features_len", len(req.Features)))

	// Check context deadline
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline)
		if remaining < 0 {
			stat.MarkErr()
			zlog.LOG.Warn("Request deadline exceeded")
			return nil, status.Error(codes.DeadlineExceeded, "request deadline exceeded")
		}
		zlog.LOG.Debug("Request deadline", zap.Duration("remaining", remaining))
	}

	// Perform ranking
	resp, err := srv.ranker.Ranking(req)
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("Ranking failed",
			zap.Error(err),
			zap.Int("entry_count", len(req.Entries)))
		return nil, status.Error(codes.Internal, fmt.Sprintf("ranking failed: %v", err))
	}

	// Log success
	zlog.LOG.Debug("gRPC ranking completed successfully",
		zap.Int("entry_count", len(resp.Entries)),
		zap.Int64("version", resp.Version),
		zap.String("model_version", resp.ModelId))

	return resp, nil
}

// ==============================================================================
// Health Check (gRPC)
// ==============================================================================

// Check implements the gRPC health check protocol
//
// Returns SERVING status if the service is healthy.
func (srv *Services) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	zlog.LOG.Debug("Health check requested",
		zap.String("service", req.GetService()))

	// Check if ranker is initialized
	if srv.ranker == nil {
		zlog.LOG.Warn("Health check failed: ranker not initialized")
		return &grpc_health_v1.HealthCheckResponse{
			Status: grpc_health_v1.HealthCheckResponse_NOT_SERVING,
		}, nil
	}

	// Service is healthy
	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

// Watch implements the gRPC health check watch protocol
//
// Currently not implemented (returns nil).
func (srv *Services) Watch(req *grpc_health_v1.HealthCheckRequest, stream grpc_health_v1.Health_WatchServer) error {
	zlog.LOG.Debug("Health watch requested",
		zap.String("service", req.GetService()))

	// TODO: Implement health status streaming
	// For now, just send initial status and return
	if srv.ranker == nil {
		return stream.Send(&grpc_health_v1.HealthCheckResponse{
			Status: grpc_health_v1.HealthCheckResponse_NOT_SERVING,
		})
	}

	return stream.Send(&grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	})
}
