package preprocess

/*
#cgo CFLAGS: -I${SRCDIR}/../third/sample-luban/include
#cgo darwin,amd64 LDFLAGS: 	-lstdc++ -lm -lpthread -L/usr/local/lib /usr/local/lib/liblua.a ${SRCDIR}/../third/lib/darwin/amd64/libluban_static.a ${SRCDIR}/../third/lib/darwin/amd64/libsample_luban_static.a
#cgo darwin,arm64 LDFLAGS: -lstdc++ -lm -lpthread  -L/usr/local/lib /usr/local/lib/liblua.a  ${SRCDIR}/../third/lib/darwin/arm64/libluban_static.a ${SRCDIR}/../third/lib/darwin/arm64/libsample_luban_static.a
#cgo linux,amd64 LDFLAGS: -lm -lpthread  -L/usr/local/lib /usr/local/lib/liblua.a  ${SRCDIR}/../third/lib/linux/amd64/libluban_static.a ${SRCDIR}/../third/lib/linux/amd64/libsample_luban_static.a
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "c_sample.h"
*/
import "C"
import (
	"os"
	"path/filepath"
	"reflect"
	"unsafe"

	"github.com/bytedance/sonic"
	"github.com/uopensail/longmen/config"
	"github.com/uopensail/ulib/commonconfig"
	"github.com/uopensail/ulib/finder"
	"github.com/uopensail/ulib/sample"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type SampleToolKitWrapper struct {
	utils.Reference
	cPtr unsafe.Pointer
}

func NewSampleToolKitWrapper(luaPluginFilePath string) *SampleToolKitWrapper {
	fielPathC := C.CString(luaPluginFilePath)
	defer C.free(unsafe.Pointer(fielPathC))
	cPtr := C.sample_luban_new_toolkit(fielPathC)
	toolkit := &SampleToolKitWrapper{
		cPtr: cPtr,
	}
	toolkit.CloseHandler = func() {
		if cPtr != nil {
			C.sample_luban_delete_toolkit(cPtr)
		}
	}
	return toolkit
}

func convertFeature(outJsonData []byte) *sample.MutableFeatures {
	feats := sample.NewMutableFeatures()
	err := sonic.Unmarshal(outJsonData, &feats)
	if err != nil {
		return nil
	}
	return feats
}

func (toolkit *SampleToolKitWrapper) ProcessUser(pool *PoolWrapper, userFeatureJson []byte) *sample.MutableFeatures {

	if len(userFeatureJson) == 0 {
		return nil
	}
	outC := C.sample_luban_process_user(toolkit.cPtr, pool.cPtr, (*C.char)(unsafe.Pointer(&userFeatureJson[0])), C.int(len(userFeatureJson)))
	if outC != nil {
		defer C.free(unsafe.Pointer(outC))
	} else {
		return nil
	}
	outLen := C.strlen(outC)
	slice := &reflect.SliceHeader{Data: uintptr(unsafe.Pointer(outC)), Len: int(outLen), Cap: int(outLen)}

	outJsonData := *(*[]byte)(unsafe.Pointer(slice))

	return convertFeature(outJsonData)
}

func (toolkit *SampleToolKitWrapper) ProcessItem(pool *PoolWrapper, itemID string) *sample.MutableFeatures {

	if len(itemID) == 0 {
		return nil
	}
	itemIDC := C.CString(itemID)
	defer C.free(unsafe.Pointer(itemIDC))

	outC := C.sample_luban_process_item(toolkit.cPtr, pool.cPtr, itemIDC, C.int(len(itemID)))
	if outC != nil {
		defer C.free(unsafe.Pointer(outC))
	} else {
		return nil
	}
	outLen := C.strlen(outC)
	slice := &reflect.SliceHeader{Data: uintptr(unsafe.Pointer(outC)), Len: int(outLen), Cap: int(outLen)}

	outJsonData := *(*[]byte)(unsafe.Pointer(slice))

	return convertFeature(outJsonData)
}

func downloadFile(dw finder.IFinder, dwCfg commonconfig.DownloaderConfig) (int64, error) {

	size, err := dw.Download(dwCfg.SourcePath, dwCfg.LocalPath)
	if err != nil {
		return 0, err
	}
	return size, err
}

type PoolWrapper struct {
	poolFileInfos []config.PoolConfig
	cPtr          unsafe.Pointer
}

// fileExists checks if a file exists and is not a directory before we
// try using it to prevent further errors.
func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if err == nil {
		return !info.IsDir() // Ensure the returned file info is not a directory
	}
	if os.IsNotExist(err) {
		return false
	}
	return true // returns true if error is not nil and it's not a "not exists" error
}

func converDownloadFileInfo(envCfg config.EnvConfig, pool config.PoolConfig) commonconfig.DownloaderConfig {
	localDir := filepath.Join(envCfg.WorkDir, "pools")
	os.MkdirAll(localDir, os.ModePerm)
	localPath := filepath.Join(localDir, pool.Name)
	dwCfg := commonconfig.DownloaderConfig{
		FinderConfig: envCfg.Finder,
		SourcePath:   pool.Path,
		LocalPath:    localPath,
	}
	return dwCfg

}
func NewPoolWrapper(envCfg config.EnvConfig, pools []config.PoolConfig) *PoolWrapper {
	fileInfos := make([]commonconfig.DownloaderConfig, len(pools))

	for i := 0; i < len(fileInfos); i++ {
		fileInfos[i] = converDownloadFileInfo(envCfg, pools[i])
		dw := finder.GetFinder(&fileInfos[i].FinderConfig)
		//Download
		_, err := downloadFile(dw, fileInfos[i])
		if err != nil {
			zlog.LOG.Error("GetUpdateFileJob error", zap.String("source", fileInfos[i].SourcePath), zap.Error(err))
			panic(err)
		}
	}

	// 创建C字符串数组
	cStrings := make([]*C.char, len(fileInfos))
	for i, s := range fileInfos {
		cStrings[i] = C.CString(s.LocalPath)
		defer C.free(unsafe.Pointer(cStrings[i])) // 记得释放内存
	}

	cPtr := C.sample_luban_new_pool_getter((**C.char)(unsafe.Pointer(&cStrings[0])), C.int(len(fileInfos)))
	pw := &PoolWrapper{
		poolFileInfos: pools,
	}
	pw.cPtr = cPtr

	return pw
}

func (pool *PoolWrapper) Close() {
	if pool.cPtr != nil {
		C.sample_luban_delete_pool_getter(pool.cPtr)
	}
}

func (pool *PoolWrapper) GetUpdateFileJob(envCfg config.EnvConfig, pools []config.PoolConfig) func() {

	jobs := make([]func(), 0, len(pools))
	for i := 0; i < len(pools); i++ {
		index := i
		newPoolInfo := pools[index]
		if newPoolInfo.Version == pool.poolFileInfos[index].Version {
			continue
		}
		newInfo := converDownloadFileInfo(envCfg, pools[index])
		job := func() {
			dw := finder.GetFinder(&newInfo.FinderConfig)
			//Download
			_, err := downloadFile(dw, newInfo)
			if err != nil {
				zlog.LOG.Error("GetUpdateFileJob error", zap.String("source", newInfo.SourcePath), zap.Error(err))
				return
			}

			cString := C.CString(newInfo.LocalPath)
			defer C.free(unsafe.Pointer(cString))
			C.sample_luban_update_pool(pool.cPtr, C.int(index), cString)
			pool.poolFileInfos[index] = newPoolInfo
			zlog.LOG.Info("sample_luban_update_pool Success", zap.String("source", newInfo.SourcePath),
				zap.String("localpath", newInfo.LocalPath))
		}
		jobs = append(jobs, job)
	}
	if len(jobs) > 0 {
		return func() {
			for i := 0; i < len(jobs); i++ {
				jobs[i]()
			}
		}
	}
	return nil
}
