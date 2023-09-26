package model

import (
	"longmen/api"
	"longmen/config"
	"longmen/wrapper"
	"sync/atomic"
	"time"

	"github.com/uopensail/ulib/prome"
)

type Instance struct {
	w   *wrapper.Wrapper
	ref int64
}

func NewInstance(poolConf *config.PoolConfigure, modelConf *config.ModelConfigure) *Instance {
	stat := prome.NewStat("NewInstance")
	defer stat.End()
	return &Instance{
		w: wrapper.NewWrapper(
			modelConf,
			poolConf,
		),
		ref: 0,
	}
}

func (ins *Instance) Release() {
	stat := prome.NewStat("Instance.Release")
	defer stat.End()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for atomic.LoadInt64(&ins.ref) != 0 {
		<-ticker.C
	}
	ins.w.Release()
}

func (ins *Instance) Rank(r *api.Request) *api.Response {
	stat := prome.NewStat("ModelInstance.Rank")
	defer stat.End()
	atomic.AddInt64(&ins.ref, 1)
	defer atomic.AddInt64(&ins.ref, -1)
	return ins.w.Rank(r)
}
