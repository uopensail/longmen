package wrapper

/*
#cgo darwin,amd64 pkg-config: ${SRCDIR}/../third/longmen-darwin-amd64.pc
#cgo darwin,arm64 pkg-config: ${SRCDIR}/../third/longmen-darwin-arm64.pc
#cgo linux,amd64 pkg-config: ${SRCDIR}/../third/longmen-linux-amd64.pc
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
)

type Wrapper struct {
	utils.Reference
	Ptr unsafe.Pointer
}

func NewWrapper(poolPath, keyField, lubanCfgPath, modelPath string) *Wrapper {
	model := C.longmen_new_model((*C.char)(unsafe.Pointer(&s2b(poolPath)[0])), C.int(len(poolPath)),
		(*C.char)(unsafe.Pointer(&s2b(keyField)[0])), C.int(len(keyField)),
		(*C.char)(unsafe.Pointer(&s2b(lubanCfgPath)[0])), C.int(len(lubanCfgPath)),
		(*C.char)(unsafe.Pointer(&s2b(modelPath)[0])), C.int(len(modelPath)))
	w := &Wrapper{
		Ptr: model,
	}

	w.CloseHandler = func() {
		if w.Ptr != nil {
			C.longmen_del_model(w.Ptr)
			w.Ptr = nil
		}
	}
	return w
}

func (w *Wrapper) Close() {
	w.Reference.LazyFree(1)
}

func (w *Wrapper) Rank(userFeatureJson string, itemIds []string) []float32 {
	stat := prome.NewStat("Wrapper.Rank")
	defer stat.End()
	w.Retain()
	defer w.Release()

	items := make([]*C.char, len(itemIds))
	lens := make([]int, len(itemIds))
	for i := 0; i < len(itemIds); i++ {
		items[i] = (*C.char)(unsafe.Pointer(&s2b(itemIds[i])[0]))
		lens[i] = len(itemIds[i])
	}

	scores := make([]float32, len(itemIds))
	C.longmen_forward(w.Ptr, (*C.char)(unsafe.Pointer(&s2b(userFeatureJson)[0])),
		C.int(len(userFeatureJson)), unsafe.Pointer(&items[0]), unsafe.Pointer(&lens[0]),
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
