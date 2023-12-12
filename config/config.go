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
	Version string `json:"version" toml:"version" yaml:"version"`
}

type ModelConfig struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Kit     string `json:"kit" toml:"kit" yaml:"kit"`
	Lua     string `json:"lua" toml:"lua" yaml:"lua"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type PoolModelConfig struct {
	ModelConfig `json:"model" toml:"model" yaml:"model"`
	PoolConfig  `json:"pool" toml:"pool" yaml:"pool"`
}
type RemoteConfig struct {
	Pool  string `json:"pool" toml:"pool" yaml:"pool"`
	Model string `json:"model" toml:"model" yaml:"model"`
}
type LocalConfig struct {
	PoolModelFile string `json:"config_file" toml:"config_file" yaml:"config_file"`
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	EnvConfig                 `json:"env" toml:"env" yaml:"env"`
	Mode                      string `json:"mode" toml:"mode" yaml:"mode"`

	RemoteConfig `json:"remote" toml:"remote" yaml:"remote"`
	LocalConfig  `json:"local" toml:"local" yaml:"local"`
}

func (conf *AppConfig) GetRemoteConfig() (*PoolModelConfig, error) {
	stat := prome.NewStat("AppConfig.GetRemoteConfig")
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

	runtimeViper2 := viper.New()
	runtimeViper2.AddRemoteProvider("etcd3",
		strings.Join(conf.EtcdConfig.Endpoints, ","),
		fmt.Sprintf(MODEL_KEY_FORMAT, conf.Model))
	runtimeViper2.SetConfigType("json")
	err = runtimeViper2.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return nil, err
	}
	modelConf := &ModelConfig{}
	err = runtimeViper.Unmarshal(modelConf)

	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper unmarshal Model config error", zap.Error(err))
		return nil, err
	}

	return &PoolModelConfig{
		PoolConfig:  *poolConf,
		ModelConfig: *modelConf,
	}, err
}

func (conf *AppConfig) GetLocalConfig() (*PoolModelConfig, error) {
	stat := prome.NewStat("AppConfig.GetLocalConfig")
	defer stat.End()
	runtimeViper := viper.New()
	fd, err := os.Open(conf.LocalConfig.PoolModelFile)
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("open  config error", zap.Error(err))
		return nil, err
	}

	defer fd.Close()
	runtimeViper.ReadConfig(fd)
	poolModelConf := &PoolModelConfig{}
	err = runtimeViper.Unmarshal(poolModelConf)
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return nil, err
	}

	return poolModelConf, err
}

func (conf *AppConfig) GetPoolModelConfig() (*PoolModelConfig, error) {
	stat := prome.NewStat("AppConfig.GetModelConfig")
	defer stat.End()
	if conf.Mode == "local" {
		return conf.GetLocalConfig()
	} else {
		return conf.GetRemoteConfig()
	}

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
