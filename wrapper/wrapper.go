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
	"fmt"
	"longmen/api"
	"longmen/config"
	"reflect"
	"unsafe"

	"github.com/uopensail/ulib/prome"
)

type Wrapper struct {
	Ptr   unsafe.Pointer
	MConf *config.ModelConfigure
	PConf *config.PoolConfigure
}

func NewWrapper(mConf *config.ModelConfigure, pConf *config.PoolConfigure) *Wrapper {
	model := C.longmen_new_model((*C.char)(unsafe.Pointer(&s2b(pConf.Path)[0])), C.int(len(pConf.Path)),
		(*C.char)(unsafe.Pointer(&s2b(pConf.Key)[0])), C.int(len(pConf.Key)),
		(*C.char)(unsafe.Pointer(&s2b(mConf.Kit)[0])), C.int(len(mConf.Kit)),
		(*C.char)(unsafe.Pointer(&s2b(mConf.Path)[0])), C.int(len(mConf.Path)))
	return &Wrapper{
		Ptr:   model,
		MConf: mConf,
		PConf: pConf,
	}
}

func (w *Wrapper) Release() {
	if w.Ptr != nil {
		C.longmen_del_model(w.Ptr)
		w.Ptr = nil
	}
}

func (w *Wrapper) Rank(r *api.Request) *api.Response {
	stat := prome.NewStat("Wrapper.Rank")
	defer stat.End()

	items := make([]*C.char, len(r.Records))
	lens := make([]int, len(r.Records))
	for i := 0; i < len(r.Records); i++ {
		items[i] = (*C.char)(unsafe.Pointer(&s2b(r.Records[i].Id)[0]))
		lens[i] = len(r.Records[i].Id)
	}

	fmt.Printf("%v\n", lens)

	scores := make([]float32, len(r.Records))
	C.longmen_forward(w.Ptr, (*C.char)(unsafe.Pointer(&s2b(r.UserFeatures)[0])),
		C.int(len(r.UserFeatures)), unsafe.Pointer(&items[0]), unsafe.Pointer(&lens[0]),
		C.int(len(r.Records)), (*C.float)(unsafe.Pointer(&scores[0])))

	for i := 0; i < len(r.Records); i++ {
		r.Records[i].Score = scores[i]
	}

	resp := &api.Response{
		Status:  0,
		UserId:  r.UserId,
		Records: r.Records,
		Extras: map[string]string{
			"mVer": w.MConf.Version,
			"pVer": w.PConf.Version,
		},
	}
	stat.SetCounter(len(r.Records))
	return resp
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
