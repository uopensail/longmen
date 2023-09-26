package model

import (
	"longmen/api"
	"longmen/config"
	"sync/atomic"
	"time"
	"unsafe"

	_ "github.com/spf13/viper/remote"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

type Manager struct {
	status int32
	ins    *Instance
}

func (mgr *Manager) Rank(r *api.Request) (*api.Response, error) {
	stat := prome.NewStat("Manager.Rank")
	defer stat.End()
	ins := mgr.ins
	return ins.Rank(r), nil
}

func (mgr *Manager) Close() {
	atomic.StoreInt32(&mgr.status, 0)
	time.Sleep(time.Second * 3)
	mgr.ins.Release()
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

		mconf, err := config.AppConf.GetModelConfig()
		if err != nil {
			zlog.LOG.Error("Manager.GetModelConfig", zap.Error(err))
			continue
		}
		pconf, err := config.AppConf.GetPoolConfig()
		if err != nil {
			zlog.LOG.Error("Manager.GetPoolConfig", zap.Error(err))
			continue
		}
		var update bool
		if mgr.ins.w.MConf.Version != mconf.Version {
			update = true
		}

		if mgr.ins.w.PConf.Version != pconf.Version {
			update = true
		}

		if !update {
			continue
		}

		ins := NewInstance(pconf, mconf)
		atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.ins)), unsafe.Pointer(mgr.ins), unsafe.Pointer(ins))
	}
}

func newManager() (*Manager, error) {
	mconf, err := config.AppConf.GetModelConfig()
	if err != nil {
		zlog.LOG.Error("Manager.GetModelConfig", zap.Error(err))
		return nil, err
	}
	pconf, err := config.AppConf.GetPoolConfig()
	if err != nil {
		zlog.LOG.Error("Manager.GetPoolConfig", zap.Error(err))
		return nil, err
	}

	mgr := &Manager{
		ins:    NewInstance(pconf, mconf),
		status: 0,
	}

	go mgr.run()

	return mgr, nil
}

var managerInstance *Manager

func Init() {
	var err error
	managerInstance, err = newManager()
	if err != nil {
		panic(err)
	}
}

func Close() {
	managerInstance.Close()
}

func Rank(r *api.Request) (*api.Response, error) {
	return managerInstance.Rank(r)
}
