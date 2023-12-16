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
	"math/rand"
	"runtime"
	"sync"
	"unsafe"

	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
)

type jobResult struct {
	rows unsafe.Pointer
}

type toolkitJob struct {
	pool          *PoolWrapper
	userFeatrJson []byte
	onDone        func(feat *jobResult)
}

type toolkitWorker struct {
	toolkit    *SampleLubanToolKitWrapper
	jobChannel chan toolkitJob
}

func newtoolkitWorker(luaFilePath, lubanFilePath string, chanSize int) *toolkitWorker {
	w := &toolkitWorker{
		toolkit:    NewSampleLubanToolKitWrapper(luaFilePath, lubanFilePath),
		jobChannel: make(chan toolkitJob, chanSize),
	}
	go w.doLoop()
	return w
}

func (worker *toolkitWorker) do(job toolkitJob) {
	var ret jobResult
	defer job.onDone(&ret)
	rows := worker.toolkit.ProcessUser(job.pool, job.userFeatrJson)
	ret.rows = rows

}
func (worker *toolkitWorker) doLoop() {
	zlog.LOG.Warn("work loop")
	for job := range worker.jobChannel {
		worker.do(job)
	}
	zlog.LOG.Warn("work loop Done")
}

func (worker *toolkitWorker) submit(job toolkitJob) {
	worker.jobChannel <- job
}

func (worker *toolkitWorker) close() {
	close(worker.jobChannel)
	worker.toolkit.LazyFree(3) //delay release
}

type PreProcessToolKit struct {
	utils.Reference
	workers []*toolkitWorker
}

func NewLuaToolKit(luaPath, lubanPath string) *PreProcessToolKit {

	chanSize := 10000
	num := runtime.NumCPU()
	workers := make([]*toolkitWorker, num)

	for i := 0; i < num; i++ {
		workers[i] = newtoolkitWorker(luaPath, lubanPath, chanSize)
	}
	tw := &PreProcessToolKit{
		workers: workers,
	}
	tw.CloseHandler = func() {
		for i := 0; i < len(tw.workers); i++ {
			tw.workers[i].close()
		}
	}
	return tw
}

func (toolkit *PreProcessToolKit) Close() {
	toolkit.LazyFree(3)
}

func (toolkit *PreProcessToolKit) getWorker() *toolkitWorker {
	i := rand.Intn(10000) % len(toolkit.workers)
	return toolkit.workers[i]
}

func (toolkit *PreProcessToolKit) ProcessUser(pool *PoolWrapper, userFeatureJson []byte) unsafe.Pointer {

	worker := toolkit.getWorker()
	wg := sync.WaitGroup{}
	ret := &jobResult{}
	job := toolkitJob{
		pool:          pool,
		userFeatrJson: userFeatureJson,
		onDone: func(ret *jobResult) {
			defer wg.Done()
			ret.rows = ret.rows

		},
	}

	wg.Add(1)
	worker.submit(job)
	wg.Wait()
	return ret.rows
}

type SampleLubanToolKitWrapper struct {
	utils.Reference
	cPtr unsafe.Pointer
}

func NewSampleLubanToolKitWrapper(luaPluginFilePath string, lubanFilePath string) *SampleLubanToolKitWrapper {
	fielPathC := C.CString(luaPluginFilePath)
	defer C.free(unsafe.Pointer(fielPathC))
	lubanFilePathC := C.CString(lubanFilePath)
	defer C.free(unsafe.Pointer(lubanFilePathC))

	cPtr := C.new_sample_luban_toolkit(fielPathC, lubanFilePathC)
	toolkit := &SampleLubanToolKitWrapper{
		cPtr: cPtr,
	}
	toolkit.CloseHandler = func() {
		if cPtr != nil {
			C.delete_sample_luban_toolkit(cPtr)
		}
	}
	return toolkit
}

func (toolkit *SampleLubanToolKitWrapper) ProcessUser(pool *PoolWrapper, userFeatureJson []byte) unsafe.Pointer {

	if len(userFeatureJson) == 0 {
		return nil
	}
	outC := C.sample_luban_new_user_rows(toolkit.cPtr, pool.cPtr, (*C.char)(unsafe.Pointer(&userFeatureJson[0])), C.int(len(userFeatureJson)))

	return outC
}

func ReleaseLubanRows(rowsPtr unsafe.Pointer) {
	C.sample_luban_delete_user_rows(rowsPtr)
}
