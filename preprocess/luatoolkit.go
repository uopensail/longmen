package preprocess

import (
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/uopensail/longmen/config"
	"github.com/uopensail/ulib/commonconfig"
	"github.com/uopensail/ulib/finder"
	"github.com/uopensail/ulib/sample"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type toolkitJob struct {
	getProcessUserArgs func() (*PoolWrapper, []byte)
	getProcessItemArgs func() (*PoolWrapper, string)
	onDone             func(feat *sample.MutableFeatures)
}

type toolkitWorker struct {
	toolkit    *SampleToolKitWrapper
	jobChannel chan toolkitJob
}

func newtoolkitWorker(luaFilePath string, chanSize int) *toolkitWorker {
	w := &toolkitWorker{
		toolkit:    NewSampleToolKitWrapper(luaFilePath),
		jobChannel: make(chan toolkitJob, chanSize),
	}
	go w.doLoop()
	return w
}

func (worker *toolkitWorker) do(job toolkitJob) {
	if job.getProcessUserArgs != nil {
		var feat sample.MutableFeatures
		defer job.onDone(&feat)
		pool, userFeatJson := job.getProcessUserArgs()
		ret := worker.toolkit.ProcessUser(pool, userFeatJson)
		feat = *ret
	}

	if job.getProcessItemArgs != nil {
		var feat sample.MutableFeatures
		defer job.onDone(&feat)
		pool, itemID := job.getProcessItemArgs()
		ret := worker.toolkit.ProcessItem(pool, itemID)
		feat = *ret

	}

}
func (worker *toolkitWorker) doLoop() {
	zlog.LOG.Warn("work loop")
	for job := range worker.jobChannel {
		worker.do(job)
	}
	zlog.LOG.Warn("work loop Done")
}

func (worker *toolkitWorker) submit(job toolkitJob) {
	worker.jobChannel <- job
}

func (worker *toolkitWorker) close() {
	close(worker.jobChannel)
	worker.toolkit.LazyFree(3) //delay release
}

type LuaToolKit struct {
	modelCfg config.ModelConfig
	utils.Reference
	workers []*toolkitWorker
}

func NewLuaToolKit(envCfg config.EnvConfig, modelCfg config.ModelConfig) *LuaToolKit {
	localDir := filepath.Join(envCfg.WorkDir, "lua")
	os.MkdirAll(localDir, os.ModePerm)
	fileName := filepath.Base(modelCfg.Lua)
	localPath := filepath.Join(localDir, fileName)
	dwCfg := commonconfig.DownloaderConfig{
		FinderConfig: envCfg.Finder,
		SourcePath:   modelCfg.Lua,
		LocalPath:    localPath,
	}

	chanSize := 10000
	num := runtime.NumCPU()
	workers := make([]*toolkitWorker, num)
	dw := finder.GetFinder(&envCfg.Finder)
	_, err := downloadFile(dw, dwCfg)
	if err != nil {
		zlog.LOG.Error("NewLuaToolKit error", zap.String("source", dwCfg.SourcePath), zap.String("local", dwCfg.LocalPath), zap.Error(err))
		panic(err)
	}

	for i := 0; i < num; i++ {
		workers[i] = newtoolkitWorker(dwCfg.LocalPath, chanSize)
	}
	tw := &LuaToolKit{
		workers:  workers,
		modelCfg: modelCfg,
	}
	tw.CloseHandler = func() {
		for i := 0; i < len(tw.workers); i++ {
			tw.workers[i].close()
		}
	}
	return tw
}

func (toolkit *LuaToolKit) Close() {
	toolkit.LazyFree(3)
}

func (toolkit *LuaToolKit) getWorker() *toolkitWorker {
	i := rand.Intn(10000) % len(toolkit.workers)
	return toolkit.workers[i]
}

func (toolkit *LuaToolKit) processUser(pool *PoolWrapper, userFeatureJson []byte) *sample.MutableFeatures {

	worker := toolkit.getWorker()
	wg := sync.WaitGroup{}
	var ret *sample.MutableFeatures
	job := toolkitJob{
		getProcessUserArgs: func() (*PoolWrapper, []byte) {
			return pool, userFeatureJson
		},
		onDone: func(feat *sample.MutableFeatures) {
			defer wg.Done()
			ret = feat

		},
	}

	wg.Add(1)
	worker.submit(job)
	wg.Wait()
	return ret
}

func (toolkit *LuaToolKit) processItem(pool *PoolWrapper, itemID string) *sample.MutableFeatures {

	worker := toolkit.getWorker()
	wg := sync.WaitGroup{}
	var ret *sample.MutableFeatures

	job := toolkitJob{
		getProcessItemArgs: func() (*PoolWrapper, string) {
			return pool, itemID
		},
		onDone: func(feat *sample.MutableFeatures) {
			defer wg.Done()
			ret = feat
		},
	}
	wg.Add(1)
	worker.submit(job)
	wg.Wait()
	return ret
}
