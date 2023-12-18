package preprocess

/*
#cgo CFLAGS: -I${SRCDIR}/../third/sample-luban/include
#cgo darwin,amd64 LDFLAGS: 	-lstdc++ -lm -lpthread -L/usr/local/lib /usr/local/lib/liblua.a ${SRCDIR}/../third/lib/darwin/amd64/libluban_static.a ${SRCDIR}/../third/lib/darwin/amd64/libsample_luban_static.a
#cgo darwin,arm64 LDFLAGS: -lstdc++ -lm -lpthread  -L/usr/local/lib /usr/local/lib/liblua.a  ${SRCDIR}/../third/lib/darwin/arm64/libluban_static.a ${SRCDIR}/../third/lib/darwin/arm64/libsample_luban_static.a
#cgo linux,amd64 LDFLAGS: -lstdc++ -lm -lpthread  -L/usr/local/lib /usr/local/lib/liblua.a  ${SRCDIR}/../third/lib/linux/amd64/libluban_static.a ${SRCDIR}/../third/lib/linux/amd64/libsample_luban_static.a
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "c_sample.h"
*/
import "C"
import (
	"os"
	"unsafe"

	"github.com/uopensail/ulib/commonconfig"
	"github.com/uopensail/ulib/finder"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

func downloadFile(dw finder.IFinder, dwCfg commonconfig.DownloaderConfig) (int64, error) {

	size, err := dw.Download(dwCfg.SourcePath, dwCfg.LocalPath)
	if err != nil {
		return 0, err
	}
	return size, err
}

type PoolWrapper struct {
	cPtr unsafe.Pointer
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

func NewPoolWrapper(files []string) *PoolWrapper {

	// 创建C字符串数组
	cStrings := make([]*C.char, len(files))
	for i, s := range files {
		cStrings[i] = C.CString(s)
		defer C.free(unsafe.Pointer(cStrings[i])) // 记得释放内存
	}

	cPtr := C.sample_luban_new_pool_getter((**C.char)(unsafe.Pointer(&cStrings[0])), C.int(len(files)))
	pw := &PoolWrapper{
		cPtr: cPtr,
	}

	return pw
}

func (pool *PoolWrapper) Close() {
	if pool.cPtr != nil {
		C.sample_luban_delete_pool_getter(pool.cPtr)
	}
}

func (pool *PoolWrapper) GetUpdateFileJob(files []string) func() error {

	jobs := make([]func() error, 0, len(files))
	for i := 0; i < len(files); i++ {
		file := files[i]
		index := i
		if fileExists(file) {
			job := func() error {
				cString := C.CString(file)
				defer C.free(unsafe.Pointer(cString))
				C.sample_luban_update_pool(pool.cPtr, C.int(index), cString)

				zlog.LOG.Info("sample_luban_update_pool Success",
					zap.String("localpath", file))
				return nil
			}
			jobs = append(jobs, job)
		}

	}
	if len(jobs) > 0 {
		return func() error {
			for i := 0; i < len(jobs); i++ {
				err := jobs[i]()
				if err != nil {
					return err
				}
			}
			return nil
		}
	}
	return nil
}
