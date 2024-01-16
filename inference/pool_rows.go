package inference

/*
#cgo CFLAGS: -I${SRCDIR}/../third/longmen/include
#cgo darwin,amd64 LDFLAGS: ${SRCDIR}/../third/lib/darwin/amd64/liblongmen_static.a ${SRCDIR}/../third/lib/darwin/amd64/libluban_static.a ${SRCDIR}/../third/lib/darwin/amd64/libsample_luban_static.a /usr/local/lib/liblua.a -L/usr/local/lib -lstdc++ -ldl -lm  -lc10 -ltorch_cpu -ltcmalloc  -lpthread
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/../third/lib/darwin/arm64/liblongmen_static.a ${SRCDIR}/../third/lib/darwin/arm64/libluban_static.a ${SRCDIR}/../third/lib/darwin/arm64/libsample_luban_static.a /usr/local/lib/liblua.a -L/usr/local/lib  -lstdc++ -ldl -lm  -lc10 -ltorch_cpu -ltcmalloc  -lpthread
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/../third/lib/linux/amd64/liblongmen_static.a ${SRCDIR}/../third/lib/linux/amd64/libluban_static.a ${SRCDIR}/../third/lib/linux/amd64/libsample_luban_static.a /usr/local/lib/liblua.a -L/usr/local/lib -lstdc++ -ldl -lm  -lc10 -ltorch_cpu -ltcmalloc -lpthread
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "longmen.h"
*/
import "C"
import (
	"unsafe"

	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type PoolRowsCache struct {
	utils.Reference
	Ptr unsafe.Pointer
}

func NewPoolRowsCache(poolFile, luaFile, lubanFile string, modelPtr unsafe.Pointer) *PoolRowsCache {
	poolFileC := C.CString(poolFile)
	defer C.free(unsafe.Pointer(poolFileC))
	luaFileC := C.CString(luaFile)
	defer C.free(unsafe.Pointer(luaFileC))
	lubanFileC := C.CString(lubanFile)
	defer C.free(unsafe.Pointer(lubanFileC))

	cppPtr := C.longmen_new_pool_rows(poolFileC, luaFileC, lubanFileC, modelPtr)
	w := &PoolRowsCache{
		Ptr: cppPtr,
	}

	w.CloseHandler = func() {
		if w.Ptr != nil {
			zlog.LOG.Info("C.longmen_new_pool_rows",
				zap.String("PoolFile", poolFile), zap.String("luaFile", luaFile),
				zap.String("lubanFile", lubanFile))

			C.longmen_delete_pool_rows(w.Ptr)
			w.Ptr = nil
		}
	}
	return w
}

func (w *PoolRowsCache) Close() {
	w.Reference.LazyFree(1)
}
