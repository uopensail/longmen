package ranker

/*
#cgo CFLAGS: -I/usr/local/include
#cgo darwin LDFLAGS: -L/usr/local/lib -L/usr/local/onnxruntime/lib -llongmen -lonnxruntime -Wl,-rpath,/usr/local/lib -Wl,-rpath,/usr/local/onnxruntime/lib -lstdc++ -lpthread
#include <stdlib.h>
#include "/usr/local/include/longmen/longmen.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/uopensail/longmen/api"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

const (
	// checkInterval defines how often to check for new model versions
	checkInterval = 300 * time.Second

	// successMarker is the filename that indicates a version is ready
	successMarker = "SUCCESS"

	// itemsFilename is the name of the feature pool file
	itemsFilename = "items.json"

	// metaFilename is the name of the model metadata file
	metaFilename = "meta.json"
)

// ==============================================================================
// Meta - Model Metadata
// ==============================================================================

// Node represents a model input/output node configuration
type Node struct {
	Name  string `json:"name"`  // Node name in the ONNX graph
	Shape []int  `json:"shape"` // Tensor shape (first dim is batch)
	Type  int    `json:"dtype"` // Data type (1=float32, 0=int64)
}

// Meta contains model metadata including output configuration
type Meta struct {
	Outputs []Node `json:"outputs"` // Output node configurations
	Version string `json:"version"` // Model version identifier
	Width   []int  `json:"-"`       // Computed feature width per output (excluding batch)
}

// NewMeta loads and parses model metadata from file
//
// The metadata file should contain JSON with model configuration including
// output nodes, shapes, and version information.
//
// Panics if file cannot be read or parsed.
func NewMeta(metaPath string) *Meta {
	stat := prome.NewStat("NewMeta")
	defer stat.End()

	zlog.LOG.Info("Loading model metadata", zap.String("path", metaPath))

	// Read metadata file
	data, err := os.ReadFile(metaPath)
	if err != nil {
		zlog.LOG.Error("Failed to read metadata file",
			zap.String("path", metaPath),
			zap.Error(err))
		panic(fmt.Sprintf("failed to read metadata: %v", err))
	}

	// Parse JSON
	var meta Meta
	if err = json.Unmarshal(data, &meta); err != nil {
		zlog.LOG.Error("Failed to parse metadata JSON",
			zap.String("path", metaPath),
			zap.Error(err))
		panic(fmt.Sprintf("failed to parse metadata: %v", err))
	}

	// Compute feature width for each output (product of dims excluding batch)
	meta.Width = make([]int, len(meta.Outputs))
	for i, node := range meta.Outputs {
		width := 1
		for j := 1; j < len(node.Shape); j++ {
			width *= node.Shape[j]
		}
		meta.Width[i] = width

		zlog.LOG.Debug("Output node configuration",
			zap.Int("index", i),
			zap.String("name", node.Name),
			zap.Ints("shape", node.Shape),
			zap.Int("width", width))
	}

	zlog.LOG.Info("Model metadata loaded successfully",
		zap.String("version", meta.Version),
		zap.Int("output_count", len(meta.Outputs)))

	return &meta
}

// ==============================================================================
// Monitor - Feature Pool Version Monitor
// ==============================================================================

// Monitor watches a directory for new feature pool versions and hot-swaps them
//
// The monitor periodically scans the work directory for timestamped subdirectories
// containing a SUCCESS marker file. When a newer version is found, it triggers
// a hot-swap of the feature pool in the model.
type Monitor struct {
	workDir    string         // Directory to monitor for new versions
	stopChan   chan struct{}  // Channel to signal monitor shutdown
	model      unsafe.Pointer // Pointer to C model instance
	lastUpdate int64          // Timestamp of last successful update (atomic)
}

// NewMonitor creates a new feature pool monitor
//
// The workDir should contain timestamped subdirectories (Unix timestamps as names)
// with feature pool files and a SUCCESS marker when ready.
func NewMonitor(workDir string, model unsafe.Pointer) (*Monitor, error) {
	// Validate work directory exists
	if _, err := os.Stat(workDir); os.IsNotExist(err) {
		zlog.LOG.Error("Work directory does not exist",
			zap.String("path", workDir),
			zap.Error(err))
		return nil, fmt.Errorf("work directory does not exist: %s", workDir)
	}

	if model == nil {
		zlog.LOG.Error("Model pointer is nil")
		return nil, fmt.Errorf("model pointer cannot be nil")
	}

	zlog.LOG.Info("Monitor created", zap.String("workdir", workDir))

	return &Monitor{
		workDir:    workDir,
		stopChan:   make(chan struct{}),
		model:      model,
		lastUpdate: 0,
	}, nil
}

// check scans for new feature pool versions and triggers hot-swap if found
func (m *Monitor) check() {
	// Read directory entries
	entries, err := os.ReadDir(m.workDir)
	if err != nil {
		zlog.LOG.Error("Failed to read work directory",
			zap.String("path", m.workDir),
			zap.Error(err))
		return
	}

	var maxTimestamp int64
	var maxTimestampDir string

	// Find the newest valid version
	for _, entry := range entries {
		// Skip non-directories
		if !entry.IsDir() {
			continue
		}

		// Parse directory name as timestamp
		timestamp, err := strconv.ParseInt(entry.Name(), 10, 64)
		if err != nil {
			// Skip non-numeric directory names
			continue
		}

		// Check if version is ready (SUCCESS marker exists)
		successFilePath := filepath.Join(m.workDir, entry.Name(), successMarker)
		if _, err := os.Stat(successFilePath); err == nil {
			if timestamp > maxTimestamp {
				maxTimestamp = timestamp
				maxTimestampDir = filepath.Join(m.workDir, entry.Name())
			}
		}
	}

	// No valid version found
	if maxTimestampDir == "" {
		zlog.LOG.Debug("No valid feature pool version found")
		return
	}

	// Check if this version is newer than current
	currentVersion := atomic.LoadInt64(&m.lastUpdate)
	if currentVersion >= maxTimestamp {
		zlog.LOG.Debug("Current version is up to date",
			zap.Int64("current", currentVersion),
			zap.Int64("found", maxTimestamp))
		return
	}

	zlog.LOG.Info("Found new feature pool version",
		zap.Int64("old_version", currentVersion),
		zap.Int64("new_version", maxTimestamp),
		zap.String("path", maxTimestampDir))

	// Prepare path to items file
	itemsPath := filepath.Join(maxTimestampDir, itemsFilename)

	// Verify items file exists
	if _, err := os.Stat(itemsPath); err != nil {
		zlog.LOG.Error("Items file not found",
			zap.String("path", itemsPath),
			zap.Error(err))
		return
	}

	// Convert path to C string
	cPath := C.CString(itemsPath)
	defer C.free(unsafe.Pointer(cPath))

	// Update version atomically before reflush (optimistic)
	atomic.StoreInt64(&m.lastUpdate, maxTimestamp)

	// Trigger hot-swap in C model
	C.longmen_reflush(
		m.model,
		cPath,
		C.int32_t(len(itemsPath)),
		C.int64_t(maxTimestamp),
	)

	zlog.LOG.Info("Feature pool refreshed successfully",
		zap.Int64("version", maxTimestamp),
		zap.String("path", itemsPath))
}

// Start begins monitoring for new feature pool versions
//
// Performs an immediate check, then continues checking at regular intervals
// until Stop() is called.
func (m *Monitor) Start() error {
	zlog.LOG.Info("Starting feature pool monitor",
		zap.String("workdir", m.workDir),
		zap.Duration("interval", checkInterval))

	// Perform initial check
	m.check()

	// Start background monitoring goroutine
	go func() {
		ticker := time.NewTicker(checkInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.check()

			case <-m.stopChan:
				zlog.LOG.Info("Feature pool monitor stopped")
				return
			}
		}
	}()

	return nil
}

// Stop gracefully shuts down the monitor
func (m *Monitor) Stop() {
	zlog.LOG.Info("Stopping feature pool monitor")
	m.stopChan <- struct{}{}
	close(m.stopChan)
}

// GetCurrentVersion returns the currently loaded version timestamp
func (m *Monitor) GetCurrentVersion() int64 {
	return atomic.LoadInt64(&m.lastUpdate)
}

// ==============================================================================
// Ranker - Main Inference Engine
// ==============================================================================

// Ranker provides high-level interface for model inference
//
// Manages the C model instance, metadata, and feature pool monitoring.
// Thread-safe for concurrent inference requests.
type Ranker struct {
	ptr     unsafe.Pointer // Pointer to C model instance
	meta    *Meta          // Model metadata
	monitor *Monitor       // Feature pool monitor
}

// NewRanker creates a new ranker instance
//
// Parameters:
//   - workdir: Directory containing model files (model.onnx, meta.json, features.json)
//   - itemDir: Directory to monitor for feature pool versions
//
// Panics if initialization fails.
func NewRanker(workdir, itemDir string) *Ranker {
	zlog.LOG.Info("Creating ranker",
		zap.String("workdir", workdir),
		zap.String("itemdir", itemDir))

	// Validate directories exist
	if _, err := os.Stat(workdir); os.IsNotExist(err) {
		zlog.LOG.Error("Work directory does not exist",
			zap.String("path", workdir))
		panic(fmt.Sprintf("workdir does not exist: %s", workdir))
	}

	if _, err := os.Stat(itemDir); os.IsNotExist(err) {
		zlog.LOG.Error("Item directory does not exist",
			zap.String("path", itemDir))
		panic(fmt.Sprintf("itemdir does not exist: %s", itemDir))
	}

	// Create C model instance
	cWorkdir := C.CString(workdir)
	model := C.longmen_create(cWorkdir, C.int32_t(len(workdir)))
	C.free(unsafe.Pointer(cWorkdir))

	if model == nil {
		zlog.LOG.Error("Failed to create LongMen model",
			zap.String("workdir", workdir))
		panic("longmen_create returned nil")
	}

	zlog.LOG.Info("LongMen model created successfully")

	// Load model metadata
	metaPath := filepath.Join(workdir, metaFilename)
	meta := NewMeta(metaPath)

	// Create and start feature pool monitor
	monitor, err := NewMonitor(itemDir, model)
	if err != nil {
		C.longmen_release(model)
		zlog.LOG.Error("Failed to create monitor", zap.Error(err))
		panic(fmt.Sprintf("failed to create monitor: %v", err))
	}

	if err := monitor.Start(); err != nil {
		C.longmen_release(model)
		zlog.LOG.Error("Failed to start monitor", zap.Error(err))
		panic(fmt.Sprintf("failed to start monitor: %v", err))
	}

	zlog.LOG.Info("Ranker created successfully",
		zap.String("model_version", meta.Version),
		zap.Int64("pool_version", monitor.GetCurrentVersion()))

	return &Ranker{
		ptr:     model,
		meta:    meta,
		monitor: monitor,
	}
}

// Close releases all resources
//
// Should be called when the ranker is no longer needed.
// Not safe to use ranker after calling Close().
func (r *Ranker) Close() {
	zlog.LOG.Info("Closing ranker")

	// Stop monitor first
	if r.monitor != nil {
		r.monitor.Stop()
	}

	// Release C model
	if r.ptr != nil {
		C.longmen_release(r.ptr)
		r.ptr = nil
	}

	zlog.LOG.Info("Ranker closed successfully")
}

// Ranking performs batch inference on the given request
//
// Processes user features and item IDs to produce ranking scores.
// Thread-safe and can be called concurrently.
//
// Returns:
//   - Response with scores for each item
//   - Error if inference fails
func (r *Ranker) Ranking(req *api.Request) (*api.Response, error) {
	stat := prome.NewStat("Ranker.Ranking")
	defer stat.End()

	batch := len(req.Entries)

	// Validate request
	if batch == 0 {
		zlog.LOG.Warn("Empty request entries")
		return &api.Response{Status: 0}, nil
	}

	if len(req.Features) == 0 {
		zlog.LOG.Warn("Empty user features")
		return nil, fmt.Errorf("user features cannot be empty")
	}

	stat.SetCounter(batch)

	zlog.LOG.Debug("Starting inference",
		zap.Int("batch_size", batch),
		zap.Int("features_len", len(req.Features)))

	// Pin memory to prevent Go GC from moving data during C call
	var pinner runtime.Pinner
	defer pinner.Unpin()

	// Prepare user features
	featuresPtr := unsafe.Pointer(unsafe.StringData(req.Features))
	pinner.Pin(featuresPtr)

	// Prepare item IDs and lengths
	items := make([]*C.char, batch)
	lens := make([]C.size_t, batch)

	for i := range batch {
		if len(req.Entries[i].Id) == 0 {
			zlog.LOG.Warn("Empty item ID", zap.Int("index", i))
			return nil, fmt.Errorf("item ID at index %d is empty", i)
		}

		idPtr := unsafe.Pointer(unsafe.StringData(req.Entries[i].Id))
		pinner.Pin(idPtr)
		items[i] = (*C.char)(idPtr)
		lens[i] = C.size_t(len(req.Entries[i].Id))
	}

	itemsPtr := unsafe.Pointer(&items[0])
	lensPtr := unsafe.Pointer(&lens[0])
	pinner.Pin(itemsPtr)
	pinner.Pin(lensPtr)

	// Prepare output buffers
	outputCount := len(r.meta.Outputs)
	scores := make([][]float32, outputCount)
	scoresPtr := make([]unsafe.Pointer, outputCount)

	for i, width := range r.meta.Width {
		size := width * batch
		scores[i] = make([]float32, size)
		scoresPtr[i] = unsafe.Pointer(&scores[i][0])
		pinner.Pin(scoresPtr[i])
	}

	pinner.Pin(&scoresPtr[0])

	var version int64

	// Call C inference function
	status := C.longmen_forward(
		r.ptr,
		(*C.char)(featuresPtr),
		C.int32_t(len(req.Features)),
		itemsPtr,
		lensPtr,
		C.int32_t(batch),
		unsafe.Pointer(&scoresPtr[0]),
		(*C.int64_t)(unsafe.Pointer(&version)),
	)

	// Check inference status
	if status != 0 {
		stat.MarkErr()
		zlog.LOG.Error("Inference failed",
			zap.Int32("status", int32(status)),
			zap.Int("batch_size", batch))
		return nil, fmt.Errorf("inference failed with status: %d", status)
	}

	// Build response
	resp := &api.Response{
		Status:  int32(status),
		Version: version,
		ModelId: r.meta.Version,
		Entries: make([]*api.Entry, batch),
	}

	for i := range batch {
		resp.Entries[i] = &api.Entry{
			Id:     req.Entries[i].Id,
			Scores: make([]*api.FloatArray, outputCount),
		}

		for n, width := range r.meta.Width {
			startIdx := i * width
			endIdx := (i + 1) * width
			resp.Entries[i].Scores[n] = &api.FloatArray{
				Values: scores[n][startIdx:endIdx],
			}
		}
	}

	zlog.LOG.Debug("Inference completed successfully",
		zap.Int("batch_size", batch),
		zap.Int64("pool_version", version),
		zap.String("model_version", r.meta.Version))

	return resp, nil
}
