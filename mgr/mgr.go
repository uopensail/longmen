package mgr

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/uopensail/longmen/config"
	"github.com/uopensail/longmen/preprocess"

	_ "github.com/spf13/viper/remote"
	"github.com/uopensail/longmen/wrapper"
	"github.com/uopensail/ulib/finder"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type Manager struct {
	ins             *wrapper.Wrapper
	preprocesser    *preprocess.PreProcesser
	curShortPoolCfg config.PoolConfig
	curLongPooCfg   config.PoolConfig
	curModelCfg     config.ModelConfig
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
func (mgr *Manager) downloadFile(envCfg config.EnvConfig, src, dst string) error {
	dw := finder.GetFinder(&envCfg.Finder)
	_, err := dw.Download(src, dst)
	return err
}
func getPath(workDir, dir, src string) string {
	poolDir := filepath.Join(workDir, dir)
	os.MkdirAll(poolDir, os.ModePerm)
	return filepath.Join(poolDir, filepath.Base(src))
}

// Do not modify the execution order
func (mgr *Manager) loadAllJob(envCfg config.EnvConfig) func() {
	shortPoolCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolShortFilePath)
	if err != nil {
		return nil
	}

	modelCfg, err := config.AppConfigInstance.GetModelConfig()
	if err != nil {
		return nil
	}
	jobs := make([]func(), 0, 2)

	if mgr.ins == nil || mgr.curModelCfg.Version != modelCfg.Version || mgr.curShortPoolCfg.Version != shortPoolCfg.Version {
		job := func() {
			poolPath := getPath(envCfg.WorkDir, "pool", shortPoolCfg.Path)
			err := mgr.downloadFile(envCfg, shortPoolCfg.Path, poolPath)
			if err != nil {
				return
			}
			modelPath := getPath(envCfg.WorkDir, "model", modelCfg.Path)
			err = mgr.downloadFile(envCfg, modelCfg.Path, modelPath)
			if err != nil {
				return
			}
			lubanPath := getPath(envCfg.WorkDir, "model", modelCfg.Kit)
			err = mgr.downloadFile(envCfg, modelCfg.Kit, lubanPath)
			if err != nil {
				return
			}

			luaPath := getPath(envCfg.WorkDir, "model", modelCfg.Lua)
			err = mgr.downloadFile(envCfg, modelCfg.Lua, luaPath)
			if err != nil {
				return
			}
			old := mgr.getInfer()
			ins := wrapper.NewWrapper(poolPath, luaPath, lubanPath, modelPath)
			if ins != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.ins)), unsafe.Pointer(ins))
				mgr.curShortPoolCfg = *shortPoolCfg
				mgr.curModelCfg = *modelCfg
			}
			if old != nil {
				old.Close()
			}

		}
		jobs = append(jobs, job)
	}

	if mgr.preprocesser == nil {
		longPoolCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolLongFilePath)
		if err != nil {
			return nil
		}
		jobs = append(jobs, func() {
			mgr.preprocesser = preprocess.NewPreProcesser(envCfg, []config.PoolConfig{
				*shortPoolCfg, *longPoolCfg,
			}, *modelCfg)
			mgr.curLongPooCfg = *longPoolCfg
		})
	} else {
		longPoolCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolLongFilePath)
		if err != nil {
			zlog.LOG.Error("config.GetPoolConfig", zap.String("PoolLongFilePath", config.AppConfigInstance.PoolLongFilePath), zap.Error(err))
		} else {
			jobs = append(jobs, func() {
				job := mgr.preprocesser.GetUpdateJob(envCfg, []config.PoolConfig{
					*shortPoolCfg, *longPoolCfg,
				}, *modelCfg)
				if job != nil {
					job()
					mgr.curLongPooCfg = *longPoolCfg
				}
			})

		}

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

func (mgr *Manager) Rank(userFeatureJson string, itemIds []string) ([]float32, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()
	infer := mgr.getInfer()
	return infer.Rank(userFeatureJson, itemIds), nil
}

var MgrIns Manager
