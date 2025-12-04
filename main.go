package main

import (
	"flag"

	_ "net/http/pprof"

	"github.com/uopensail/example-service/boot"
	"github.com/uopensail/longmen/config"
	"github.com/uopensail/longmen/services"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

func main() {
	// Parse command line flags
	configFilePath := flag.String("config", "conf/config.toml", "Path to configuration file")
	logDir := flag.String("log", "./logs", "Path to log directory")
	flag.Parse()

	// Initialize configuration
	config.AppConfigInstance.Init(*configFilePath)

	// Initialize logger
	zlog.InitLogger(config.AppConfigInstance.ProjectName, config.AppConfigInstance.Debug, *logDir)

	// Validate project name
	if len(config.AppConfigInstance.ProjectName) <= 0 {
		zlog.LOG.Fatal("config.ProjectName is empty")
	}

	// Create service instance
	srv := services.NewServices()
	if srv == nil {
		zlog.LOG.Fatal("Failed to create service instance")
	}

	zlog.LOG.Info("Service instance created successfully")

	// Start server (blocking call, handles signals internally)
	zlog.LOG.Info("Starting server",
		zap.String("project", config.AppConfigInstance.ProjectName),
		zap.String("config", *configFilePath),
		zap.String("logdir", *logDir))

	boot.Load(config.AppConfigInstance.ServerConfig, *logDir, srv)

	zlog.LOG.Info("Server stopped")
}
