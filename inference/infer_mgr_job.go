package inference

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/uopensail/longmen/config"
	"github.com/uopensail/longmen/inference/preprocess"

	"github.com/codeclysm/extract/v3"

	"github.com/uopensail/ulib/finder"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type ResourceVersion struct {
	BigPoolVersion   config.PoolConfig
	SmallPoolVersion config.PoolConfig
	ModelVesion      config.ModelConfig
}

func (mgr *InferMgr) TickerLoadJob(envCfg config.EnvConfig, jobUtil *utils.MetuxJobUtil) {
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
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		ticker := time.NewTicker(time.Minute * 5)
		defer ticker.Stop()
		for {
			<-ticker.C
			jobs := mgr.loadAllJob(envCfg)
			if len(jobs) <= 0 {
				continue
			}
			job := func() {
				hasErr := false

				for i := 0; i < len(jobs); i++ {
					job := jobs[i]
					if job != nil {
						err := job()
						if err != nil {
							zlog.LOG.Error("job load error", zap.Error(err))
							hasErr = true
						}
					}
				}
				if hasErr {
					//clean dir
					zlog.LOG.Info("os.RemoveAll", zap.String("path", envCfg.WorkDir))
					os.RemoveAll(envCfg.WorkDir)
				}
			}

			jobUtil.TryRun(job)
		}
	}()

}

func downloadFile(envCfg config.EnvConfig, src, dst string) (int64, error) {
	dw := finder.GetFinder(&envCfg.Finder)
	return dw.Download(src, dst)
}

func (mgr *InferMgr) loadAllJob(envCfg config.EnvConfig) []func() error {
	os.MkdirAll(envCfg.WorkDir, os.ModePerm)
	smallPoolRemoteCfg, err := config.AppConfigIns.ReadSmallPoolVersion()
	if err != nil {
		return nil
	}

	modelRemoteCfg, err := config.AppConfigIns.ReadModelVersion()
	if err != nil {
		return nil
	}
	bigPoolRemoteCfg, err := config.AppConfigIns.ReadBigPoolVersion()
	if err != nil {
		return nil
	}
	memSmallPoolCfg := mgr.ResourceVersion.SmallPoolVersion
	memBigPoolCfg := mgr.ResourceVersion.BigPoolVersion
	memModelCfg := mgr.ResourceVersion.ModelVesion

	jobs := make([]func() error, 0, 2)
	downloadResult := &struct {
		BigPoolFile   string
		SmallPoolFile string
		ModelDir      string
		ModelName     string
	}{
		BigPoolFile:   memBigPoolCfg.Path,
		SmallPoolFile: memSmallPoolCfg.Path,
		ModelDir:      memModelCfg.Path,
		ModelName:     memModelCfg.Name,
	}
	//拷贝到工作目录
	if memSmallPoolCfg.Version != smallPoolRemoteCfg.Version {
		jobs = append(jobs, func() error {
			stat := prome.NewStat("download.small_pool")
			defer stat.End()
			modelDir := filepath.Join(envCfg.WorkDir, "pool")
			os.MkdirAll(modelDir, os.ModePerm)
			dwFilePath := filepath.Join(modelDir, filepath.Base(smallPoolRemoteCfg.Path))
			dwSize, err := downloadFile(envCfg, smallPoolRemoteCfg.Path, dwFilePath)
			if err != nil {
				stat.MarkErr()
				zlog.LOG.Warn("download file error", zap.String("source", smallPoolRemoteCfg.Path), zap.Error(err))
				return err
			}
			stat.SetCounter(int(dwSize))
			zlog.LOG.Info("download file success", zap.String("source", smallPoolRemoteCfg.Path), zap.String("dst", dwFilePath))

			zlog.LOG.Info("os.Remove", zap.String("path", memSmallPoolCfg.Path),
				zap.String("new_path", dwFilePath))
			os.Remove(memSmallPoolCfg.Path)
			downloadResult.SmallPoolFile = dwFilePath

			return nil
		})
	}

	if memBigPoolCfg.Version != bigPoolRemoteCfg.Version {
		jobs = append(jobs, func() error {
			stat := prome.NewStat("download.big_pool")
			defer stat.End()
			modelDir := filepath.Join(envCfg.WorkDir, "pool")
			os.MkdirAll(modelDir, os.ModePerm)
			dwFilePath := filepath.Join(modelDir, filepath.Base(bigPoolRemoteCfg.Path))
			dwSize, err := downloadFile(envCfg, bigPoolRemoteCfg.Path, dwFilePath)
			if err != nil {
				stat.MarkErr()
				zlog.LOG.Warn("download file error", zap.String("source", bigPoolRemoteCfg.Path), zap.Error(err))
				return err
			}

			zlog.LOG.Info("download file success", zap.String("source", bigPoolRemoteCfg.Path), zap.String("dst", dwFilePath))
			stat.SetCounter(int(dwSize))

			zlog.LOG.Info("os.Remove", zap.String("path", memBigPoolCfg.Path),
				zap.String("new_path", dwFilePath))
			os.Remove(memBigPoolCfg.Path)
			downloadResult.BigPoolFile = dwFilePath
			return nil
		})
	}

	if memModelCfg.Version != modelRemoteCfg.Version {
		jobs = append(jobs, func() error {
			stat := prome.NewStat("download.model")
			defer stat.End()
			modelDir := filepath.Join(envCfg.WorkDir, "model")
			os.MkdirAll(modelDir, os.ModePerm)
			//download model.tar.gz

			modelFilePath := filepath.Join(modelDir, filepath.Base(modelRemoteCfg.Path))
			dwSize, err := downloadFile(envCfg, modelRemoteCfg.Path, modelFilePath)
			if err != nil {
				stat.MarkErr()
				zlog.LOG.Warn("download file error", zap.String("source", modelRemoteCfg.Path), zap.Error(err))
				return err
			}
			stat.SetCounter(int(dwSize))
			zlog.LOG.Info("download file success", zap.String("source", modelRemoteCfg.Path),
				zap.String("dst", modelFilePath))

			// 1.解压包

			file, err := os.Open(modelFilePath)
			if err != nil {
				return err
			}
			defer file.Close()
			tarDir := modelFilePath + ".dir"
			os.MkdirAll(tarDir, os.ModePerm)
			err = extract.Gz(context.TODO(), file, tarDir, nil)

			if err != nil {
				return err
			}
			downloadResult.ModelDir = tarDir
			downloadResult.ModelName = modelRemoteCfg.Name
			return nil
		})
	}

	//加载任务
	if memSmallPoolCfg.Version == "" || memBigPoolCfg.Version == "" {
		jobs = append(jobs, func() error {
			stat := prome.NewStat("load.pool_getter")
			defer stat.End()
			old := mgr.poolGetter
			//new pool Getter
			zlog.LOG.Info("preprocess.NewPoolWrapper", zap.String("small_pool", downloadResult.SmallPoolFile),
				zap.String("big_pool", downloadResult.BigPoolFile))

			poolGet := preprocess.NewPoolWrapper([]string{
				downloadResult.SmallPoolFile, downloadResult.BigPoolFile,
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
		if memSmallPoolCfg.Version != smallPoolRemoteCfg.Version {
			files[0] = downloadResult.SmallPoolFile
		}
		if memBigPoolCfg.Version != bigPoolRemoteCfg.Version {
			files[1] = downloadResult.BigPoolFile
		}
		job := mgr.poolGetter.GetUpdateFileJob(files)
		if job != nil {
			jobs = append(jobs, job)
		}
	}

	//update model
	if memModelCfg.Version != modelRemoteCfg.Version {
		jobs = append(jobs, func() error {
			stat := prome.NewStat("load.model")
			defer stat.End()
			//reload model
			zlog.LOG.Info("NewInference Begin", zap.String("model_dir", downloadResult.ModelDir))
			modelFile := filepath.Join(downloadResult.ModelDir, "model.pt")
			metaFile := filepath.Join(downloadResult.ModelDir, "model.meta.json")
			ins := NewInference(modelFile, metaFile)
			if ins != nil {
				old := mgr.getInfer()
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.inferImpl)), unsafe.Pointer(ins))
				if old != nil {
					old.Close()
				}
			}
			zlog.LOG.Info("NewInference End", zap.String("model_dir", downloadResult.ModelDir))

			return nil
		})
		//  reload lua plugin

		jobs = append(jobs, func() error {
			stat := prome.NewStat("load.preprocess")
			defer stat.End()
			luaFile := filepath.Join(downloadResult.ModelDir, fmt.Sprintf("%s.lua", downloadResult.ModelName))
			lubanFIle := filepath.Join(downloadResult.ModelDir, "luban_config.json_new")
			zlog.LOG.Info("preprocess.NewPreProcessToolKit Begin", zap.String("lua_file", luaFile))

			preprocessToolkit := preprocess.NewPreProcessToolKit(luaFile, lubanFIle)
			old := (*preprocess.PreProcessToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit))))
			if preprocessToolkit != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit)), unsafe.Pointer(preprocessToolkit))
				if old != nil {
					old.Close()
				}
			}
			zlog.LOG.Info("preprocess.NewPreProcessToolKit End", zap.String("lua_file", luaFile))

			return nil
		})
	}
	if memModelCfg.Version != modelRemoteCfg.Version || memSmallPoolCfg.Version != smallPoolRemoteCfg.Version {
		//reload pool rows
		jobs = append(jobs, func() error {
			stat := prome.NewStat("load.pool_rows")
			defer stat.End()
			luaFile := filepath.Join(downloadResult.ModelDir, fmt.Sprintf("%s.lua", downloadResult.ModelName))
			lubanFile := filepath.Join(downloadResult.ModelDir, "luban_config.json_new")
			zlog.LOG.Info("NewPoolRowsCache")

			modelPtr := mgr.getInfer().GetTorchModel()
			poolRowsCache := NewPoolRowsCache(downloadResult.SmallPoolFile, luaFile,
				lubanFile, modelPtr)
			old := (*PoolRowsCache)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.poolRowsCache))))
			if poolRowsCache != nil {
				atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.poolRowsCache)), unsafe.Pointer(poolRowsCache))
				if old != nil {
					old.Close()
				}
			}
			zlog.LOG.Info("NewPoolRowsCache End")

			return nil
		})
	}
	if len(jobs) > 0 {
		//mark success
		jobs = append(jobs, func() error {
			stat := prome.NewStat("load.success")
			defer stat.End()

			mgr.ResourceVersion.SmallPoolVersion = config.PoolConfig{
				Name:    smallPoolRemoteCfg.Name,
				Path:    downloadResult.SmallPoolFile,
				Version: smallPoolRemoteCfg.Version,
			}

			mgr.ResourceVersion.BigPoolVersion = config.PoolConfig{
				Name:    bigPoolRemoteCfg.Name,
				Path:    downloadResult.BigPoolFile,
				Version: bigPoolRemoteCfg.Version,
			}

			mgr.ResourceVersion.ModelVesion = config.ModelConfig{
				Name:    modelRemoteCfg.Name,
				Path:    downloadResult.ModelDir,
				Version: modelRemoteCfg.Version,
			}
			zlog.LOG.Info("Reload End", zap.Any("Version", mgr.ResourceVersion))

			return nil
		})
	}
	return jobs
}
