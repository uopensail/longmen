package preprocess

import (
	"sync/atomic"
	"unsafe"

	"github.com/uopensail/longmen/config"
	"github.com/uopensail/ulib/sample"
)

type PreProcesser struct {
	pool       *PoolWrapper
	luaToolkit *LuaToolKit
}

func NewPreProcesser(envCfg config.EnvConfig, pools []config.PoolConfig, modelCfg config.ModelConfig) *PreProcesser {

	pool := NewPoolWrapper(envCfg, pools)
	lua := NewLuaToolKit(envCfg, modelCfg)

	return &PreProcesser{
		pool:       pool,
		luaToolkit: lua,
	}
}

func (p *PreProcesser) Close() {
	p.pool.Close()
	p.luaToolkit.Close()
}

func (p *PreProcesser) ProcessUser(userFeatureJson []byte) *sample.MutableFeatures {
	toolkitPorcess := (*LuaToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&p.luaToolkit))))

	toolkitPorcess.Retain()
	defer toolkitPorcess.Release()
	return toolkitPorcess.processUser(p.pool, userFeatureJson)
}

func (p *PreProcesser) ProcessItem(pluginName string, itemID string) *sample.MutableFeatures {
	toolkitPorcess := (*LuaToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&p.luaToolkit))))

	toolkitPorcess.Retain()
	defer toolkitPorcess.Release()

	return toolkitPorcess.processItem(p.pool, itemID)
}

func (p *PreProcesser) GetUpdateJob(envCfg config.EnvConfig, pools []config.PoolConfig, modelCfg config.ModelConfig) func() {

	jobs := make([]func(), 0, 2)
	job := p.pool.GetUpdateFileJob(envCfg, pools)

	if job != nil {
		jobs = append(jobs, job)
	}

	if modelCfg.Version != p.luaToolkit.modelCfg.Version {
		jobs = append(jobs, func() {
			luaToolkit := NewLuaToolKit(envCfg, modelCfg)
			old := (*LuaToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&p.luaToolkit))))
			atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&p.luaToolkit)), unsafe.Pointer(&luaToolkit))
			if old != nil {
				old.Close()
			}
		})
	}

	if len(jobs) > 0 {
		return func() {
			for i := 0; i < len(jobs); i++ {
				jobs[i]()
			}
		}
	}
	return nil
}
