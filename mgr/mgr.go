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

type ModelFileRessource struct {
	modelFilePath, luaFilePath, lubanFilePath string
}
type FileResource struct {
	shortPoolLocalMeta config.PoolConfig
	longPoolLocalMeta  config.PoolConfig
	modelLocalMeta     config.ModelConfig
}

func (res *FileResource) GetLocalShortPoolMeta() *config.PoolConfig {
	cfg := config.PoolConfig{}
	err := cfg.Init(res.shortPoolLocalMeta.Path + ".meta")
	if err != nil {
		return nil
	}
	return &cfg
}

func (res *FileResource) GetLocalLongPoolMeta() *config.PoolConfig {
	cfg := config.PoolConfig{}
	err := cfg.Init(res.longPoolLocalMeta.Path + ".meta")
	if err != nil {
		return nil
	}
	return &cfg
}

func (res *FileResource) GetLocalModelMeta() *config.ModelConfig {
	cfg := config.ModelConfig{}
	err := cfg.Init(res.modelLocalMeta.CheckpiontPath + ".meta")
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
				hasErr := false

				for i := 0; i < len(jobs); i++ {
					job := jobs[i]
					if job != nil {
						err := job()
						if err != nil {
							hasErr = true
						}
					}
				}
				if hasErr {
					//clean dir
					os.RemoveAll(envCfg.WorkDir)
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
	os.MkdirAll(envCfg.WorkDir, os.ModePerm)
	shortPoolRemoteCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolShortFilePath)
	if err != nil {
		return nil
	}

	modelRemoteCfg, err := config.AppConfigInstance.GetModelConfig()
	if err != nil {
		return nil
	}
	longPoolRemoteCfg, err := config.AppConfigInstance.GetPoolConfig(config.AppConfigInstance.PoolLongFilePath)
	if err != nil {
		return nil
	}
	localLongPoolCfg := mgr.FileResource.GetLocalLongPoolMeta()
	localShortPoolCfg := mgr.FileResource.GetLocalShortPoolMeta()
	localModelCfg := mgr.FileResource.GetLocalModelMeta()

	jobs := make([]func() error, 0, 2)

	if localShortPoolCfg == nil || localShortPoolCfg.Version != shortPoolRemoteCfg.Version {
		jobs = append(jobs, func() error {
			dwFilePath := getPath(envCfg.WorkDir, "pool", shortPoolRemoteCfg.Path)
			err := mgr.downloadFile(envCfg, shortPoolRemoteCfg.Path, dwFilePath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", shortPoolRemoteCfg.Path), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", shortPoolRemoteCfg.Path), zap.String("dst", dwFilePath))

			newLocalConfig := config.PoolConfig{
				Name:    shortPoolRemoteCfg.Name,
				Path:    dwFilePath,
				Version: shortPoolRemoteCfg.Version,
			}
			mgr.FileResource.shortPoolLocalMeta = newLocalConfig
			newLocalConfig.Dump(mgr.FileResource.shortPoolLocalMeta.Path + ".meta")
			return nil
		})
	}

	if localModelCfg == nil || localModelCfg.Version != modelRemoteCfg.Version {
		jobs = append(jobs, func() error {
			modelDir := filepath.Join(envCfg.WorkDir, "model")
			os.MkdirAll(modelDir, os.ModePerm)

			//download model.pt
			modelFilePath := filepath.Join(modelDir, filepath.Base(modelRemoteCfg.CheckpiontPath))
			err := mgr.downloadFile(envCfg, modelRemoteCfg.CheckpiontPath, modelFilePath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", modelRemoteCfg.CheckpiontPath), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", modelRemoteCfg.CheckpiontPath),
				zap.String("dst", modelFilePath))

			//download luban.json
			lubanPath := filepath.Join(modelDir, filepath.Base(modelRemoteCfg.Kit))
			err = mgr.downloadFile(envCfg, modelRemoteCfg.Kit, lubanPath)
			if err != nil {
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", modelRemoteCfg.Kit),
				zap.String("dst", lubanPath))
			//download model.lua
			luaPath := filepath.Join(modelDir, filepath.Base(modelRemoteCfg.Lua))
			err = mgr.downloadFile(envCfg, modelRemoteCfg.Lua, luaPath)
			if err != nil {
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", modelRemoteCfg.Lua), zap.String("dst", luaPath))

			newLocalConfig := config.ModelConfig{
				CheckpiontPath: modelFilePath,
				Lua:            luaPath,
				Kit:            lubanPath,
				Version:        shortPoolRemoteCfg.Version,
			}
			mgr.modelLocalMeta = newLocalConfig
			newLocalConfig.Dump(mgr.FileResource.modelLocalMeta.CheckpiontPath + ".meta")
			return nil
		})
	}

	if localLongPoolCfg == nil || localLongPoolCfg.Version != longPoolRemoteCfg.Version {
		jobs = append(jobs, func() error {
			//download pool.json
			longPoolPath := getPath(envCfg.WorkDir, "pool", longPoolRemoteCfg.Path)
			err := mgr.downloadFile(envCfg, longPoolRemoteCfg.Path, longPoolPath)
			if err != nil {
				zlog.LOG.Warn("download file   error", zap.String("source", longPoolRemoteCfg.Path), zap.Error(err))
				return err
			}
			zlog.LOG.Info("download file success", zap.String("source", longPoolRemoteCfg.Path), zap.String("dst", longPoolPath))
			newLocalConfig := config.PoolConfig{
				Name:    longPoolRemoteCfg.Name,
				Path:    longPoolPath,
				Version: longPoolRemoteCfg.Version,
			}
			mgr.longPoolLocalMeta = newLocalConfig
			newLocalConfig.Dump(mgr.longPoolLocalMeta.Path + ".meta")
			return nil
		})
	}

	if localLongPoolCfg == nil || localShortPoolCfg == nil {
		jobs = append(jobs, func() error {
			old := mgr.poolGetter
			//new pool Getter
			poolGet := preprocess.NewPoolWrapper([]string{
				mgr.shortPoolLocalMeta.Path, mgr.longPoolLocalMeta.Path,
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
		files := make([]string, 2)
		if localShortPoolCfg == nil || localShortPoolCfg.Version != shortPoolRemoteCfg.Version {
			files[0] = mgr.FileResource.shortPoolLocalMeta.Path
		}
		if localLongPoolCfg == nil || localLongPoolCfg.Version != longPoolRemoteCfg.Version {
			files[1] = mgr.FileResource.longPoolLocalMeta.Path
		}
		job := mgr.poolGetter.GetUpdateFileJob(files)
		if job != nil {
			jobs = append(jobs, job)
		}
	}

	//update model
	if localModelCfg == nil || localModelCfg.Version != modelRemoteCfg.Version {
		jobs = append(jobs, func() error {
			old := mgr.getInfer()
			//reload model
			ins := wrapper.NewWrapper(mgr.FileResource.shortPoolLocalMeta.Path,
				mgr.FileResource.modelLocalMeta.Lua,
				mgr.FileResource.modelLocalMeta.Kit,
				mgr.FileResource.modelLocalMeta.CheckpiontPath)
			if ins != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.modelIns)), unsafe.Pointer(ins))
				if old != nil {
					old.Close()
				}
			}

			return nil
		})
	}

	//  reload lua plugin

	if localModelCfg == nil || localModelCfg.Version != modelRemoteCfg.Version {
		jobs = append(jobs, func() error {
			preprocessToolkit := preprocess.NewPreProcessToolKit(modelRemoteCfg.Lua, modelRemoteCfg.Kit)
			old := (*preprocess.PreProcessToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit))))
			if preprocessToolkit != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit)), unsafe.Pointer(preprocessToolkit))
				if old != nil {
					old.Close()
				}
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

func (mgr *Manager) Rank(userFeatureJson string, itemIds []string) ([]float32, map[string]string, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()

	featData := []byte(userFeatureJson)
	if mgr.preprocessToolkit == nil {
		return nil, nil, errors.New("preprocess empty")
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
	versionMap := map[string]string{
		"version.pool":  mgr.shortPoolLocalMeta.Version,
		"version.model": mgr.modelLocalMeta.Version,
	}
	return score, versionMap, nil
}

var MgrIns Manager
