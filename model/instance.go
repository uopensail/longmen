package model

import (
	"longmen/api"
	"longmen/config"
	"longmen/wrapper"
	"sync/atomic"
	"time"

	"github.com/uopensail/ulib/prome"
)

type PoolInstance struct {
	pool *wrapper.PoolWrapper
	ref  int64
}

func NewPoolInstance(conf *config.PoolConfigure) *PoolInstance {
	stat := prome.NewStat("NewPoolInstance")
	defer stat.End()
	return &PoolInstance{
		pool: wrapper.NewPoolWrapper(conf.Path, conf.Version, conf.Key),
		ref:  0,
	}
}

func (ins *PoolInstance) Release() {
	stat := prome.NewStat("PoolInstance.Release")
	defer stat.End()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for atomic.LoadInt64(&ins.ref) != 0 {
		<-ticker.C
	}
	ins.pool.Release()
}

func (ins *PoolInstance) Version() string {
	atomic.AddInt64(&ins.ref, 1)
	defer atomic.AddInt64(&ins.ref, -1)
	return ins.pool.Version()
}

type ModelInstance struct {
	model *wrapper.ModelWrapper
	pool  string
	ref   int64
}

func NewModelInstance(conf *config.ModelConfigure) *ModelInstance {
	stat := prome.NewStat("NewModelInstance")
	defer stat.End()
	return &ModelInstance{
		model: wrapper.NewModelWrapper(conf.Path, conf.Version, conf.Kit),
		ref:   0,
		pool:  conf.Pool,
	}
}

func (ins *ModelInstance) Version() string {
	atomic.AddInt64(&ins.ref, 1)
	defer atomic.AddInt64(&ins.ref, -1)
	return ins.model.Version()
}

func (ins *ModelInstance) Release() {
	stat := prome.NewStat("ModelInstance.Release")
	defer stat.End()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for atomic.LoadInt64(&ins.ref) != 0 {
		<-ticker.C
	}
	ins.model.Release()
}

func (ins *ModelInstance) Rank(p *PoolInstance, r *api.Request) *api.Response {
	stat := prome.NewStat("ModelInstance.Rank")
	defer stat.End()

	atomic.AddInt64(&ins.ref, 1)
	atomic.AddInt64(&p.ref, 1)
	defer atomic.AddInt64(&ins.ref, -1)
	defer atomic.AddInt64(&p.ref, -1)
	return ins.model.Rank(p.pool, r)
}
