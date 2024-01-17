package inference

/*
#cgo CFLAGS: -I${SRCDIR}/../third/longmen/include
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/../third/lib/darwin/amd64/ -llongmen_static -lluban_static -lsample_luban_static /usr/local/lib/liblua.a -L/usr/local/lib -lstdc++ -ldl -lm -lc10 -ltorch_cpu  -ltcmalloc -lpthread
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/../third/lib/darwin/arm64/ -llongmen_static -lluban_static -lsample_luban_static /usr/local/lib/liblua.a -L/usr/local/lib  -lstdc++ -ldl -lm -lc10 -ltorch_cpu -ltcmalloc  -lpthread
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../third/lib/linux/amd64/ -llongmen_static -lluban_static -lsample_luban_static /usr/local/lib/liblua.a -L/usr/local/lib -lstdc++ -ldl -lm -lc10 -ltorch_cpu -ltcmalloc  -lpthread
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "longmen.h"
*/
import "C"
import (
	"reflect"
	"unsafe"

	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type Inference struct {
	utils.Reference
	Ptr unsafe.Pointer
}

func NewInference(modeFile, modelMeta string) *Inference {

	modeFileC := C.CString(modeFile)
	defer C.free(unsafe.Pointer(modeFileC))
	modelMetaC := C.CString(modelMeta)
	defer C.free(unsafe.Pointer(modelMetaC))
	cppPtr := C.new_longmen_torch_model(modeFileC, modelMetaC)
	w := &Inference{
		Ptr: cppPtr,
	}

	w.CloseHandler = func() {
		if w.Ptr != nil {
			zlog.LOG.Info("C.delete_longmen_torch_model",
				zap.String("modelMeta", modelMeta), zap.String("modeFile", modeFile))
			C.delete_longmen_torch_model(w.Ptr)
			w.Ptr = nil
		}
	}
	return w
}

func (w *Inference) GetTorchModel() unsafe.Pointer {
	return w.Ptr
}

func (w *Inference) Close() {
	w.Reference.LazyFree(1)
}
func (w *Inference) PreProcessUserEmbedding(userFeatureRowsPtr unsafe.Pointer, poolPtr unsafe.Pointer) {
	C.longmen_user_rows_embedding_preforward(w.Ptr, userFeatureRowsPtr, poolPtr)
}
func (w *Inference) Rank(userFeatureRowsPtr unsafe.Pointer, poolRowsPtr unsafe.Pointer, itemIds []string) []float32 {
	stat := prome.NewStat("Inference.Rank")
	defer stat.End()

	items := make([]*C.char, len(itemIds))
	lens := make([]int, len(itemIds))
	for i := 0; i < len(itemIds); i++ {
		items[i] = (*C.char)(unsafe.Pointer(&s2b(itemIds[i])[0]))
		lens[i] = len(itemIds[i])
	}

	scores := make([]float32, len(itemIds))
	C.longmen_torch_model_inference(w.Ptr, userFeatureRowsPtr, poolRowsPtr, (*C.char)(unsafe.Pointer(&items[0])), unsafe.Pointer(&lens[0]),
		C.int(len(itemIds)), (*C.float)(unsafe.Pointer(&scores[0])))

	stat.SetCounter(len(itemIds))
	return scores
}

func s2b(s string) (b []byte) {
	/* #nosec G103 */
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	/* #nosec G103 */
	sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
	bh.Data = sh.Data
	bh.Cap = sh.Len
	bh.Len = sh.Len
	return b
}
