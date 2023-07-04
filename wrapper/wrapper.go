package wrapper

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/usr/local/lob -L../lib -llongmen -Wl,-rpath,./../lib
#include "../cpp/longmen.h"
#include <stdlib.h>
*/
import "C"

import (
	"longmen/api"
	"unsafe"
)

type Model struct {
	mPtr unsafe.Pointer // model pointer
	mVer string         // model version
}

func NewModel(version, toolkit, pool, model, key string) *Model {
	toolkitStr := C.CString(toolkit)
	poolStr := C.CString(pool)
	modelStr := C.CString(model)
	keyStr := C.CString(key)
	defer C.free(unsafe.Pointer(toolkitStr))
	defer C.free(unsafe.Pointer(poolStr))
	defer C.free(unsafe.Pointer(modelStr))
	defer C.free(unsafe.Pointer(keyStr))
	return &Model{
		mPtr: C.longmen_new(toolkitStr, poolStr, modelStr, keyStr),
		mVer: version,
	}
}

func (m *Model) Relese() {
	C.longmen_release(m.mPtr)
}

func (m *Model) Version() string {
	return m.mVer
}

func (m *Model) Reload(path string) {
	pool := C.CString(path)
	defer C.free(unsafe.Pointer(pool))
	C.longmen_reload(m.mPtr, pool)
}

func (m *Model) Rank(r *api.Request) *api.Response {
	items := make([]*C.char, len(r.Records))
	for i := 0; i < len(r.Records); i++ {
		items[i] = C.CString(r.Records[i].Id)
		defer C.free(unsafe.Pointer(items[i]))
	}

	scores := make([]float32, len(r.Records))
	cstr := C.longmen_forward(m.mPtr, (*C.char)(unsafe.Pointer(&r.UserFeatures[0])),
		C.int(len(r.UserFeatures)),
		unsafe.Pointer(&items[0]),
		C.int(len(r.Records)), (*C.float)(unsafe.Pointer(&scores[0])))
	defer C.free(unsafe.Pointer(cstr))

	for i := 0; i < len(r.Records); i++ {
		r.Records[i].Score = scores[i]
	}

	resp := &api.Response{
		Status:  0,
		UserId:  r.UserId,
		Records: r.Records,
		Extras:  make(map[string]string),
	}
	resp.Extras["mId"] = r.ModelId
	resp.Extras["mVer"] = m.mVer
	resp.Extras["pVer"] = C.GoString(cstr)
	return resp
}
