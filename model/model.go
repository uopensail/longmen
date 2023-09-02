package model

import (
	"fmt"
	"longmen/api"
	"longmen/config"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spf13/viper"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

const POOL_KEY_FORMAT = "/longmen/models/%s"
const MODEL_KEY_FORMAT = "/longmen/models/%s"

type PoolWarehouse struct {
	sync.RWMutex
	ins map[string]*PoolInstance
}

func (w *PoolWarehouse) Get(key string) *PoolInstance {
	stat := prome.NewStat("PoolWarehouse.Get")
	defer stat.End()
	w.RUnlock()
	defer w.RUnlock()
	if pool, ok := w.ins[key]; ok {
		return pool
	}
	stat.MarkMiss()
	return nil
}

func (w *PoolWarehouse) Close() {
	stat := prome.NewStat("PoolWarehouse.Close")
	defer stat.End()
	w.Lock()
	defer w.Unlock()
	for _, pool := range w.ins {
		pool.Release()
	}
}

func (w *PoolWarehouse) reflush(name string) {
	stat := prome.NewStat("PoolWarehouse.reflush")
	defer stat.End()

	runtimeViper := viper.New()
	runtimeViper.AddRemoteProvider("etcd",
		strings.Join(config.AppConf.EtcdConfig.Endpoints, ","),
		fmt.Sprint(POOL_KEY_FORMAT, name))
	runtimeViper.SetConfigType("json")
	err := runtimeViper.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return
	}
	conf := &config.PoolConfigure{}
	err = runtimeViper.Unmarshal(conf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal PoolConfigure config error", zap.Error(err))
		return
	}

	pool := w.Get(name)
	if pool != nil && pool.Version() == conf.Version {
		return
	}
	newPool := NewPoolInstance(conf)
	w.Lock()
	w.ins[name] = newPool
	w.Unlock()
	if pool != nil {
		pool.Release()
	}
}

type ModelWarehouse struct {
	sync.RWMutex
	ins map[string]*ModelInstance
}

func (w *ModelWarehouse) Get(key string) *ModelInstance {
	stat := prome.NewStat("ModelWarehouse.Get")
	defer stat.End()
	w.RLock()
	defer w.RUnlock()
	if model, ok := w.ins[key]; ok {
		return model
	}
	stat.MarkMiss()
	return nil
}

func (w *ModelWarehouse) Close() {
	stat := prome.NewStat("ModelWarehouse.Close")
	defer stat.End()
	w.Lock()
	defer w.Unlock()
	for _, model := range w.ins {
		model.Release()
	}
}

func (w *ModelWarehouse) reflush(name string) {
	stat := prome.NewStat("ModelWarehouse.reflush")
	defer stat.End()

	runtimeViper := viper.New()
	runtimeViper.AddRemoteProvider("etcd",
		strings.Join(config.AppConf.Endpoints, ","),
		fmt.Sprint(MODEL_KEY_FORMAT, name))
	runtimeViper.SetConfigType("json")
	err := runtimeViper.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return
	}
	conf := &config.ModelConfigure{}
	err = runtimeViper.Unmarshal(conf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal ModelConfigure config error", zap.Error(err))
		return
	}

	model := w.Get(name)
	if model != nil && model.Version() == conf.Version {
		return
	}
	newModel := NewModelInstance(conf)
	w.Lock()
	w.ins[name] = newModel
	w.Unlock()
	if model != nil {
		model.Release()
	}
}

type Manager struct {
	status     int32
	mWarehouse *ModelWarehouse
	pWarehouse *PoolWarehouse
}

func (mgr *Manager) Rank(r *api.Request) (*api.Response, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()
	model := mgr.mWarehouse.Get(r.ModelId)
	if model == nil {
		stat.MarkErr()
		zlog.LOG.Error("model nil", zap.String("model", r.ModelId))
		return nil, fmt.Errorf("model nil: %s", r.ModelId)
	}

	pool := mgr.pWarehouse.Get(model.pool)
	if pool == nil {
		stat.MarkErr()
		zlog.LOG.Error("pool nil", zap.String("pool", model.pool))
		return nil, fmt.Errorf("pool nil: %s", model.pool)
	}
	return model.Rank(pool, r), nil
}

func (mgr *Manager) Close() {
	atomic.StoreInt32(&mgr.status, 0)
	time.Sleep(time.Second * 3)
	mgr.mWarehouse.Close()
	mgr.pWarehouse.Close()
}

func (mgr *Manager) run() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var count int64
	for atomic.LoadInt32(&mgr.status) == 1 {
		<-ticker.C
		count++
		if count%60 != 0 {
			continue
		}

		for i := 0; i < len(config.AppConf.Pools); i++ {
			go func(name string) {
				mgr.pWarehouse.reflush(name)
			}(config.AppConf.Pools[i])
		}

		for i := 0; i < len(config.AppConf.Models); i++ {
			go func(name string) {
				mgr.mWarehouse.reflush(name)
			}(config.AppConf.Models[i])
		}
	}
}

func newManager() *Manager {
	mgr := &Manager{
		mWarehouse: &ModelWarehouse{
			ins: make(map[string]*ModelInstance, len(config.AppConf.Models)),
		},
		pWarehouse: &PoolWarehouse{
			ins: make(map[string]*PoolInstance, len(config.AppConf.Pools)),
		},
	}

	for i := 0; i < len(config.AppConf.Pools); i++ {
		mgr.pWarehouse.ins[config.AppConf.Pools[i]] = nil
		mgr.pWarehouse.reflush(config.AppConf.Pools[i])
	}

	for i := 0; i < len(config.AppConf.Models); i++ {
		mgr.mWarehouse.ins[config.AppConf.Models[i]] = nil
		mgr.mWarehouse.reflush(config.AppConf.Models[i])
	}

	go mgr.run()

	return mgr
}

var managerInstance *Manager

func Init() {
	managerInstance = newManager()
}

func Close() {
	managerInstance.Close()
}

func Rank(r *api.Request) (*api.Response, error) {
	return managerInstance.Rank(r)
}
