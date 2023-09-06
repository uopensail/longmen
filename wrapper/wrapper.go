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
	"longmen/api"
	"reflect"
	"unsafe"

	"github.com/uopensail/ulib/prome"
)

type PoolWrapper struct {
	mPtr  unsafe.Pointer
	mPath string
	mVer  string
	mKey  string
}

func NewPoolWrapper(path, version, key string) *PoolWrapper {
	return &PoolWrapper{
		mPtr: C.longmen_new_pool((*C.char)(unsafe.Pointer(&s2b(path)[0])), C.int(len(path)),
			(*C.char)(unsafe.Pointer(&s2b(key)[0])), C.int(len(key))),
		mPath: path,
		mVer:  version,
		mKey:  key,
	}
}

func (pool *PoolWrapper) Release() {
	if pool.mPtr != nil {
		C.longmen_del_pool(pool.mPtr)
	}
	pool.mPtr = nil
}

func (pool *PoolWrapper) Version() string {
	return pool.mVer
}

type ModelWrapper struct {
	mPtr  unsafe.Pointer
	mPath string
	mVer  string
	mKit  string
}

func NewModelWrapper(path, version, kit string) *ModelWrapper {
	return &ModelWrapper{
		mPtr: C.longmen_new_model((*C.char)(unsafe.Pointer(&s2b(kit)[0])),
			C.int(len(kit)),
			(*C.char)(unsafe.Pointer(&s2b(path)[0])),
			C.int(len(path))),
		mPath: path,
		mVer:  version,
		mKit:  kit,
	}
}

func (m *ModelWrapper) Release() {
	if m.mPtr != nil {
		C.longmen_del_model(m.mPtr)
	}
	m.mPtr = nil
}

func (m *ModelWrapper) Version() string {
	return m.mVer
}

func (m *ModelWrapper) Rank(pool *PoolWrapper, r *api.Request) *api.Response {
	stat := prome.NewStat("ModelWrapper.Rank")
	defer stat.End()

	items := make([]*C.char, len(r.Records))
	lens := make([]int, len(r.Records))
	for i := 0; i < len(r.Records); i++ {
		items[i] = (*C.char)(unsafe.Pointer(&s2b(r.Records[i].Id)[0]))
		lens[i] = len(r.Records[i].Id)
	}

	scores := make([]float32, len(r.Records))
	C.longmen_forward(m.mPtr, pool.mPtr, (*C.char)(unsafe.Pointer(&s2b(r.UserFeatures)[0])),
		C.int(len(r.UserFeatures)), unsafe.Pointer(&items[0]), (*C.int)(unsafe.Pointer(&lens[0])),
		C.int(len(r.Records)), (*C.float)(unsafe.Pointer(&scores[0])))

	for i := 0; i < len(r.Records); i++ {
		r.Records[i].Score = scores[i]
	}

	resp := &api.Response{
		Status:  0,
		UserId:  r.UserId,
		Records: r.Records,
		Extras: map[string]string{
			"mVer": m.mVer,
			"pVer": pool.mVer,
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
