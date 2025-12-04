package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
	"github.com/uopensail/ulib/commonconfig"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

// ==============================================================================
// Configuration Structures
// ==============================================================================

// EnvConfig contains environment-specific configuration for the ranker
type EnvConfig struct {
	WorkDir string `json:"workdir" toml:"workdir" yaml:"workdir"` // Model files directory (model.onnx, meta.json, features.json)
	ItemDir string `json:"itemdir" toml:"itemdir" yaml:"itemdir"` // Feature pool versions directory
}

// Validate checks if the environment configuration is valid
func (c *EnvConfig) Validate() error {
	if c.WorkDir == "" {
		return fmt.Errorf("workdir cannot be empty")
	}

	if c.ItemDir == "" {
		return fmt.Errorf("itemdir cannot be empty")
	}

	// Check if directories exist
	if _, err := os.Stat(c.WorkDir); os.IsNotExist(err) {
		return fmt.Errorf("workdir does not exist: %s", c.WorkDir)
	}

	if _, err := os.Stat(c.ItemDir); os.IsNotExist(err) {
		return fmt.Errorf("itemdir does not exist: %s", c.ItemDir)
	}

	return nil
}

// AppConfig represents the complete application configuration
//
// Combines server configuration (ports, timeouts, etc.) with
// ranker-specific environment configuration.
type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"` // Server configuration (inherited)
	EnvConfig                 `json:"env" toml:"env" yaml:"env"`          // Environment configuration
}

// Validate checks if the application configuration is valid
func (c *AppConfig) Validate() error {
	// Validate environment config
	if err := c.EnvConfig.Validate(); err != nil {
		return fmt.Errorf("env config validation failed: %w", err)
	}

	// Validate server config (if it has a Validate method)
	// Add server config validation here if needed

	return nil
}

// String returns a string representation of the config (for logging)
func (c *AppConfig) String() string {
	return fmt.Sprintf("AppConfig{WorkDir: %s, ItemDir: %s}",
		c.WorkDir, c.ItemDir)
}

// ==============================================================================
// Global Configuration Instance
// ==============================================================================

// AppConfigInstance is the global application configuration instance
//
// Should be initialized once at application startup using Init().
var AppConfigInstance AppConfig

// ==============================================================================
// Configuration Loading
// ==============================================================================

// Init loads and parses the configuration file
//
// Supports TOML format. The configuration file should contain:
//   - [server] section with server configuration
//   - [env] section with workdir and itemdir paths
//
// Example TOML:
//
//	[server]
//	port = 8080
//	timeout = 30
//
//	[env]
//	workdir = "/path/to/model"
//	itemdir = "/path/to/items"
//
// Panics if the file cannot be read, parsed, or validation fails.
func (conf *AppConfig) Init(filePath string) {
	zlog.LOG.Info("Loading application configuration",
		zap.String("path", filePath))

	// Validate file path
	if filePath == "" {
		zlog.LOG.Error("Configuration file path is empty")
		panic("configuration file path cannot be empty")
	}

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		zlog.LOG.Error("Configuration file does not exist",
			zap.String("path", filePath),
			zap.Error(err))
		panic(fmt.Sprintf("configuration file not found: %s", filePath))
	}

	// Read configuration file
	fData, err := os.ReadFile(filePath)
	if err != nil {
		zlog.LOG.Error("Failed to read configuration file",
			zap.String("path", filePath),
			zap.Error(err))
		panic(fmt.Sprintf("failed to read config file: %v", err))
	}

	// Parse TOML
	if _, err = toml.Decode(string(fData), conf); err != nil {
		zlog.LOG.Error("Failed to parse configuration file",
			zap.String("path", filePath),
			zap.Error(err))
		panic(fmt.Sprintf("failed to parse config file: %v", err))
	}

	// Validate configuration
	if err := conf.Validate(); err != nil {
		zlog.LOG.Error("Configuration validation failed",
			zap.String("path", filePath),
			zap.Error(err))
		panic(fmt.Sprintf("invalid configuration: %v", err))
	}

	// Convert to absolute paths
	conf.WorkDir = toAbsPath(conf.WorkDir)
	conf.ItemDir = toAbsPath(conf.ItemDir)

	zlog.LOG.Info("Application configuration loaded successfully",
		zap.String("workdir", conf.WorkDir),
		zap.String("itemdir", conf.ItemDir))

	// Log configuration details (debug level)
	zlog.LOG.Debug("Configuration details",
		zap.String("config", conf.String()))
}

// toAbsPath converts a path to absolute path
func toAbsPath(path string) string {
	if path == "" {
		return path
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		zlog.LOG.Warn("Failed to convert to absolute path",
			zap.String("path", path),
			zap.Error(err))
		return path
	}

	return absPath
}
