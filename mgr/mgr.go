package mgr

import (
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/uopensail/longmen/config"

	_ "github.com/spf13/viper/remote"
	"github.com/uopensail/longmen/wrapper"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type Manager struct {
	ins    *wrapper.Wrapper
	curCfg config.PoolModelConfig
}

func (mgr *Manager) getInfer() *wrapper.Wrapper {
	ret := (*wrapper.Wrapper)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.ins))))
	return ret
}

func (mgr *Manager) Init(envCfg config.EnvConfig, jobUtil *utils.MetuxJobUtil) {

	mgr.cronJob(envCfg, jobUtil)
}

func (mgr *Manager) cronJob(envCfg config.EnvConfig, jobUtil *utils.MetuxJobUtil) {
	job := mgr.loadAllJob(envCfg)
	job()
	go func() {

		ticker := time.NewTicker(time.Minute * 5)
		defer ticker.Stop()
		for {
			<-ticker.C
			job := mgr.loadAllJob(envCfg)
			jobUtil.TryRun(job)
		}
	}()

}

// Do not modify the execution order
func (mgr *Manager) loadAllJob(envCfg config.EnvConfig) func() {
	mconf, err := config.AppConfigInstance.GetModelConfig()
	if err != nil {
		zlog.LOG.Error("Manager.GetModelConfig", zap.Error(err))
		return nil
	}
	pconf, err := config.AppConfigInstance.GetPoolConfig()
	if err != nil {
		zlog.LOG.Error("Manager.GetPoolConfig", zap.Error(err))
		return nil
	}
	update := false
	if mgr.curCfg.ModelConfig.Version != mconf.Version || mgr.curCfg.PoolConfig.Version != pconf.Version {
		update = true
	}

	if !update {
		return nil
	}
	job := func() {
		old := mgr.getInfer()
		ins := wrapper.NewWrapper(pconf.Path, pconf.Key, mconf.Kit, mconf.Path)
		if ins != nil {
			atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.ins)), unsafe.Pointer(ins))
			mgr.curCfg = config.PoolModelConfig{
				PoolConfig:  *pconf,
				ModelConfig: *mconf,
			}
			old.Close()
		}

	}
	return job
}

func (mgr *Manager) Rank(userFeatureJson string, itemIds []string) ([]float32, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()
	infer := mgr.getInfer()
	return infer.Rank(userFeatureJson, itemIds), nil
}

var MgrIns Manager
