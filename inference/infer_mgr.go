package inference

import (
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/uopensail/longmen/inference/preprocess"

	"github.com/uopensail/ulib/prome"
)

type InferMgr struct {
	ResourceVersion
	inferImpl         *Inference
	poolRowsCache     *PoolRowsCache
	poolGetter        *preprocess.PoolWrapper
	preprocessToolkit *preprocess.PreProcessToolKit
}

func (mgr *InferMgr) getInfer() *Inference {
	ret := (*Inference)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.inferImpl))))
	return ret
}

func (mgr *InferMgr) getPoolRowsCache() *PoolRowsCache {
	ret := (*PoolRowsCache)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.poolRowsCache))))
	return ret
}

func (mgr *InferMgr) preProcessUser(pool *preprocess.PoolWrapper, userFeatureJson []byte) unsafe.Pointer {
	toolkitPorcess := (*preprocess.PreProcessToolKit)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&mgr.preprocessToolkit))))
	toolkitPorcess.Retain()
	defer toolkitPorcess.Release()
	return toolkitPorcess.ProcessUser(pool, userFeatureJson)
}
func (mgr *InferMgr) UserFeatureRows(userFeatureJson string) (string, error) {
	stat := prome.NewStat("Manager.UserFeatureRows")
	defer stat.End()

	featData := []byte(userFeatureJson)
	if mgr.preprocessToolkit == nil {
		return "", errors.New("preprocess empty")
	}
	rowsPtr := mgr.preProcessUser(mgr.poolGetter, featData)

	//TODO delete
	defer preprocess.ReleaseLubanRows(rowsPtr)
	ret := preprocess.LubanRowsDumpJson(rowsPtr)
	return ret, nil
}
func (mgr *InferMgr) Rank(userFeatureJson string, itemIds []string) ([]float32, map[string]string, error) {
	stat := prome.NewStat("Manager.Rank").SetCounter(len(itemIds))
	defer stat.End()

	featData := []byte(userFeatureJson)
	if mgr.preprocessToolkit == nil {
		return nil, nil, errors.New("preprocess empty")
	}

	rowsPtr := mgr.preProcessUser(mgr.poolGetter, featData)

	//TODO delete
	defer preprocess.ReleaseLubanRows(rowsPtr)

	parallelNum := (runtime.NumCPU() / 3) * 2
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
	poolRowsCache := mgr.getPoolRowsCache()
	poolRowsCache.Retain()
	defer poolRowsCache.Release()

	//process user embedding
	infer.PreProcessUserEmbedding(rowsPtr, poolRowsCache.Ptr)
	wg := sync.WaitGroup{}
	i := 0
	score := make([]float32, itemLen)
	for ; i < itemLen; i += step {
		wg.Add(1)
		k := i + step
		if k > itemLen {
			k = itemLen
		}
		go func(begin, end int) {
			defer wg.Done()

			ret := infer.Rank(rowsPtr, poolRowsCache.Ptr, itemIds[begin:end])
			copy(score[begin:end], ret)
		}(i, k)
	}

	if i < itemLen {
		wg.Add(1)
		go func(begin, end int) {
			defer wg.Done()
			ret := infer.Rank(rowsPtr, poolRowsCache.Ptr, itemIds[begin:end])
			copy(score[begin:end], ret)
		}(i, itemLen)
	}
	wg.Wait()
	return score, nil, nil
}
