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

type EnvConfig struct {
	Finder  commonconfig.FinderConfig `json:"finder" toml:"finder"`
	WorkDir string                    `json:"work_dir" toml:"work_dir"`
}

const POOL_KEY_FORMAT = "/pools/%s"
const MODEL_KEY_FORMAT = "/longmen/models/%s"

type PoolConfig struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Key     string `json:"key" toml:"key" yaml:"key"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type ModelConfig struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Kit     string `json:"kit" toml:"kit" yaml:"kit"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type PoolModelConfig struct {
	ModelConfig `json:"model" toml:"model" yaml:"model"`
	PoolConfig  `json:"pool" toml:"pool" yaml:"pool"`
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	EnvConfig                 `json:"env" toml:"env" yaml:"env"`
	Pool                      string `json:"pool" toml:"pool" yaml:"pool"`
	Model                     string `json:"model" toml:"model" yaml:"model"`
}

func (conf *AppConfig) GetPoolConfig() (*PoolConfig, error) {
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
	poolConf := &PoolConfig{}
	err = runtimeViper.Unmarshal(poolConf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal PoolConfigure config error", zap.Error(err))
		return nil, err
	}
	return poolConf, nil
}

func (conf *AppConfig) GetModelConfig() (*ModelConfig, error) {
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
	modelConf := &ModelConfig{}
	err = runtimeViper.Unmarshal(modelConf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal ModelConfigure config error", zap.Error(err))
		return nil, err
	}
	return modelConf, nil
}

var AppConfigInstance AppConfig

func (conf *AppConfig) Init(filePath string) {
	fData, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Errorf("ioutil.ReadFile error: %s", err)
		panic(err)
	}
	_, err = toml.Decode(string(fData), conf)
	if err != nil {
		fmt.Errorf("Unmarshal error: %s", err)
		panic(err)
	}
	fmt.Printf("InitAppConfig:%v yaml:%s\n", conf, string(fData))
}
