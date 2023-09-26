package config

import (
	"fmt"
	"os"
	"strings"

	"github.com/BurntSushi/toml"
	"github.com/spf13/viper"
	"github.com/uopensail/ulib/commonconfig"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
)

const POOL_KEY_FORMAT = "/pools/%s"
const MODEL_KEY_FORMAT = "/longmen/models/%s"

type PoolConfigure struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Key     string `json:"key" toml:"key" yaml:"key"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type ModelConfigure struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Kit     string `json:"kit" toml:"kit" yaml:"kit"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	Pool                      string `json:"pool" toml:"pool" yaml:"pool"`
	Model                     string `json:"model" toml:"model" yaml:"model"`
}

func (conf *AppConfig) Init(filePath string) {
	fileData, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	if strings.HasSuffix(filePath, ".toml") {
		if _, err := toml.Decode(string(fileData), conf); err != nil {
			panic(err)
		}
	}
}

func (conf *AppConfig) GetPoolConfig() (*PoolConfigure, error) {
	stat := prome.NewStat("AppConfig.GetPoolConfig")
	defer stat.End()

	runtimeViper := viper.New()
	runtimeViper.AddRemoteProvider("etcd3",
		strings.Join(conf.EtcdConfig.Endpoints, ","),
		fmt.Sprintf(POOL_KEY_FORMAT, conf.Pool))
	runtimeViper.SetConfigType("json")
	err := runtimeViper.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return nil, err
	}
	poolConf := &PoolConfigure{}
	err = runtimeViper.Unmarshal(poolConf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal PoolConfigure config error", zap.Error(err))
		return nil, err
	}
	return poolConf, nil
}

func (conf *AppConfig) GetModelConfig() (*ModelConfigure, error) {
	stat := prome.NewStat("AppConfig.GetModelConfig")
	defer stat.End()

	runtimeViper := viper.New()
	runtimeViper.AddRemoteProvider("etcd3",
		strings.Join(conf.EtcdConfig.Endpoints, ","),
		fmt.Sprintf(MODEL_KEY_FORMAT, conf.Model))
	runtimeViper.SetConfigType("json")
	err := runtimeViper.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return nil, err
	}
	modelConf := &ModelConfigure{}
	err = runtimeViper.Unmarshal(modelConf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal ModelConfigure config error", zap.Error(err))
		return nil, err
	}
	return modelConf, nil
}

var AppConf AppConfig
