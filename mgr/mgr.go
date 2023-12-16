package mgr

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
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

type FileResource struct {
	shortPoolMetaFilePath string
	longPoolMetaFilePath  string
	modelMetaFilePath     string
	config.ModelConfig
}

func (res *FileResource) GetLocalShortPoolMeta() *config.PoolConfig {
	cfg := config.PoolConfig{}
	err := cfg.Init(res.shortPoolMetaFilePath)
	if err != nil {
		return nil
	}
	return &cfg
}

func (res *FileResource) GetLocalLongPoolMeta() *config.PoolConfig {
	cfg := config.PoolConfig{}
	err := cfg.Init(res.longPoolMetaFilePath)
	if err != nil {
		return nil
	}
	return &cfg
}

func (res *FileResource) GetLocalModelMeta() *config.ModelConfig {
	cfg := config.ModelConfig{}
	err := cfg.Init(res.modelMetaFilePath)
	if err != nil {
		return nil
	}
	return &cfg
}

type Manager struct {
	//
	FileResource
	modelIns          *wrapper.Wrapper
	poolGetter        *preprocess.PoolWrapper
	preprocessToolkit *preprocess.PreProcessToolKit
}

func (mgr *Manager) getInfer() *wrapper.Wrapper {
	ret := (*wrapper.Wrapper)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.modelIns))))
	return ret
}

func (mgr *Manager) Init(envCfg config.EnvConfig, jobUtil *utils.MetuxJobUtil) {

	mgr.cronJob(envCfg, jobUtil)
}

func (mgr *Manager) cronJob(envCfg config.EnvConfig, jobUtil *utils.MetuxJobUtil) {
	jobs := mgr.loadAllJob(envCfg)
	for i := 0; i < len(jobs); i++ {
		job := jobs[i]
		if job != nil {
			err := job()
			if err != nil {
				panic(err)
			}
		}
	}
	go func() {

		ticker := time.NewTicker(time.Minute * 5)
		defer ticker.Stop()
		for {
			<-ticker.C
			jobs := mgr.loadAllJob(envCfg)
			job := func() {
				for i := 0; i < len(jobs); i++ {
					job := jobs[i]
					if job != nil {
						job()
					}
				}
			}
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
func (mgr *Manager) loadAllJob(envCfg config.EnvConfig) []func() error {
	shortPoolCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolShortFilePath)
	if err != nil {
		return nil
	}

	modelCfg, err := config.AppConfigInstance.GetModelConfig()
	if err != nil {
		return nil
	}
	longPoolCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolLongFilePath)
	if err != nil {
		return nil
	}
	localLongPoolCfg := mgr.FileResource.GetLocalLongPoolMeta()
	localShortPoolCfg := mgr.FileResource.GetLocalShortPoolMeta()
	localModelCfg := mgr.FileResource.GetLocalModelMeta()

	jobs := make([]func() error, 0, 2)

	if localShortPoolCfg == nil || localShortPoolCfg.Version != shortPoolCfg.Version {
		jobs = append(jobs, func() error {
			dwFilePath := getPath(envCfg.WorkDir, "pool", shortPoolCfg.Path)
			err := mgr.downloadFile(envCfg, shortPoolCfg.Path, dwFilePath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", shortPoolCfg.Path), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", shortPoolCfg.Path), zap.String("dst", dwFilePath))
			mgr.FileResource.shortPoolMetaFilePath = dwFilePath
			return nil
		})
	}

	if localModelCfg == nil || localModelCfg.Version != modelCfg.Version {
		jobs = append(jobs, func() error {
			//download model.pt
			dwFilePath := getPath(envCfg.WorkDir, "model", modelCfg.Path)
			err := mgr.downloadFile(envCfg, localModelCfg.Path, dwFilePath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", localModelCfg.Path), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", localModelCfg.Path), zap.String("dst", dwFilePath))

			//download luban.json
			lubanPath := getPath(envCfg.WorkDir, "model", modelCfg.Kit)
			err = mgr.downloadFile(envCfg, modelCfg.Kit, lubanPath)
			if err != nil {
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", modelCfg.Kit), zap.String("dst", lubanPath))
			//download model.lua
			luaPath := getPath(envCfg.WorkDir, "model", modelCfg.Lua)
			err = mgr.downloadFile(envCfg, modelCfg.Lua, luaPath)
			if err != nil {
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", modelCfg.Lua), zap.String("dst", luaPath))
			mgr.FileResource.ModelConfig = *modelCfg
			mgr.FileResource.modelMetaFilePath = dwFilePath
			return nil
		})
	}

	if localLongPoolCfg == nil || localLongPoolCfg.Version != longPoolCfg.Version {
		jobs = append(jobs, func() error {
			//download pool.json
			longPoolPath := getPath(envCfg.WorkDir, "pool", longPoolCfg.Path)
			err := mgr.downloadFile(envCfg, longPoolCfg.Path, longPoolPath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", longPoolCfg.Path), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", longPoolCfg.Path), zap.String("dst", longPoolPath))
			mgr.FileResource.longPoolMetaFilePath = longPoolPath
			return nil
		})
	}

	if localLongPoolCfg == nil || localShortPoolCfg == nil {
		jobs = append(jobs, func() error {
			old := mgr.poolGetter
			//new pool Getter
			poolGet := preprocess.NewPoolWrapper(envCfg, []config.PoolConfig{
				*shortPoolCfg, *longPoolCfg,
			})
			if poolGet != nil {
				mgr.poolGetter = poolGet
				if old != nil {
					old.Close()
				}
			}
			return nil
		})
	} else {
		//reload   pool Getter
		job := mgr.poolGetter.GetUpdateFileJob(envCfg, []config.PoolConfig{
			*shortPoolCfg, *longPoolCfg,
		})
		if job != nil {
			jobs = append(jobs, job)
		}
	}

	//update model
	if localModelCfg == nil || localModelCfg.Version != modelCfg.Version {
		jobs = append(jobs, func() error {
			old := mgr.getInfer()
			//reload model
			ins := wrapper.NewWrapper(mgr.FileResource.shortPoolMetaFilePath,
				mgr.FileResource.ModelConfig.Lua, mgr.FileResource.ModelConfig.Kit, mgr.FileResource.ModelConfig.Path)
			if ins != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.modelIns)), unsafe.Pointer(ins))
			}
			if old != nil {
				old.Close()
			}
			return nil
		})
	}

	//  reload lua plugin

	if localModelCfg == nil || localModelCfg.Version != modelCfg.Version {
		jobs = append(jobs, func() error {
			preprocessToolkit := preprocess.NewLuaToolKit(modelCfg.Lua, modelCfg.Kit)
			old := (*preprocess.PreProcessToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit))))
			atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit)), unsafe.Pointer(&preprocessToolkit))
			if old != nil {
				old.Close()
			}
			return nil
		})
	}

	return jobs
}

func (mgr *Manager) preProcessUser(pool *preprocess.PoolWrapper, userFeatureJson []byte) unsafe.Pointer {
	toolkitPorcess := (*preprocess.PreProcessToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit))))

	toolkitPorcess.Retain()
	defer toolkitPorcess.Release()
	return toolkitPorcess.ProcessUser(pool, userFeatureJson)
}

func (mgr *Manager) Rank(userFeatureJson string, itemIds []string) ([]float32, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()

	featData := []byte(userFeatureJson)
	if mgr.preprocessToolkit == nil {
		return nil, errors.New("preprocess empty")
	}
	rowsPtr := mgr.preProcessUser(mgr.poolGetter, featData)

	//TODO delete
	defer preprocess.ReleaseLubanRows(rowsPtr)
	parallelNum := runtime.NumCPU()
	if parallelNum == 0 {
		parallelNum = 2
	}
	itemLen := len(itemIds)
	step := itemLen / parallelNum
	if step == 0 {
		step = 1
	}

	infer := mgr.getInfer()
	infer.Retain()
	defer infer.Release()

	wg := sync.WaitGroup{}
	i := 0
	score := make([]float32, itemLen)
	for ; i < itemLen; i += step {
		wg.Add(1)
		go func(begin, end int) {
			defer wg.Done()
			ret := infer.Rank(rowsPtr, itemIds[begin:end])
			copy(score[begin:end], ret)
		}(i, i+step)
	}

	if i < itemLen {
		wg.Add(1)
		go func(begin, end int) {
			defer wg.Done()
			ret := infer.Rank(rowsPtr, itemIds[begin:end])
			copy(score[begin:end], ret)
		}(i, itemLen)
	}
	wg.Wait()
	return score, nil
}

var MgrIns Manager
