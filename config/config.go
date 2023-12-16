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
	Name    string `json:"name" toml:"name" yaml:"name"`
	Path    string `json:"path" toml:"path" yaml:"path"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

func (conf *PoolConfig) Init(filePath string) error {
	fData, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Errorf("ioutil.ReadFile error: %s", err)
		return err
	}
	_, err = toml.Decode(string(fData), conf)
	if err != nil {
		fmt.Errorf("Unmarshal error: %s", err)
		return err
	}
	return nil
}

type ModelConfig struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Kit     string `json:"kit" toml:"kit" yaml:"kit"`
	Lua     string `json:"lua" toml:"lua" yaml:"lua"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

func (conf *ModelConfig) Init(filePath string) error {
	fData, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Errorf("ioutil.ReadFile error: %s", err)
		return err
	}
	_, err = toml.Decode(string(fData), conf)
	if err != nil {
		fmt.Errorf("Unmarshal error: %s", err)
		return err
	}
	return nil
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	EnvConfig                 `json:"env" toml:"env" yaml:"env"`
	Mode                      string `json:"mode" toml:"mode" yaml:"mode"`

	ModelFile         string `json:"model" toml:"model" yaml:"model"`
	PoolShortFilePath string `json:"pool_short" toml:"pool_short" yaml:"pool_short"`
	PoolLongFilePath  string `json:"pool_long" toml:"pool_long" yaml:"pool_long"`
}

func parseRemoteConfig(endpoints []string, etcdPath string, parseFunc func(v *viper.Viper) error) error {
	stat := prome.NewStat("getParseRemoteConfig")
	defer stat.End()
	runtimeViper := viper.New()
	runtimeViper.AddRemoteProvider("etcd3",
		strings.Join(endpoints, ","), etcdPath)
	runtimeViper.SetConfigType("json")
	err := runtimeViper.ReadRemoteConfig()
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("viper read remote config error", zap.Error(err))
		return err
	}
	if parseFunc != nil {
		err := parseFunc(runtimeViper)
		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper unmarshal PoolConfigure config error", zap.Error(err))
			return err
		}
	}

	return nil

}

func (conf *AppConfig) GetPoolConfig(poolPath string) (*PoolConfig, error) {
	stat := prome.NewStat("AppConfig.GetShortPool")
	defer stat.End()
	if conf.Mode == "local" {
		poolConf := &struct {
			Pool PoolConfig `json:"pool" toml:"pool" yaml:"pool"`
		}{}
		err := parseLocalConfig(poolPath, func(v *viper.Viper) error {
			return v.Unmarshal(poolConf)
		})

		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper read remote config error", zap.Error(err))
			return nil, err
		}
		return &poolConf.Pool, nil
	} else {
		poolConf := &PoolConfig{}
		err := parseRemoteConfig(conf.EtcdConfig.Endpoints, poolPath,
			func(v *viper.Viper) error {
				return v.Unmarshal(poolConf)
			})

		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper read remote config error", zap.Error(err))
			return nil, err
		}
		return poolConf, nil
	}
}

func (conf *AppConfig) GetModelConfig() (*ModelConfig, error) {
	stat := prome.NewStat("AppConfig.GetShortPool")
	defer stat.End()
	if conf.Mode == "local" {
		modelConf := &struct {
			Model ModelConfig `json:"model" toml:"model" yaml:"model"`
		}{}
		err := parseLocalConfig(conf.ModelFile, func(v *viper.Viper) error {
			return v.Unmarshal(modelConf)
		})

		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper read remote config error", zap.Error(err))
			return nil, err
		}
		return &modelConf.Model, nil
	} else {
		modelConf := &ModelConfig{}
		err := parseRemoteConfig(conf.EtcdConfig.Endpoints, conf.ModelFile,
			func(v *viper.Viper) error {

				return v.Unmarshal(modelConf)
			})
		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper read remote config error", zap.Error(err))
			return nil, err
		}
		return modelConf, nil
	}

}

func parseLocalConfig(filePath string, parseFunc func(v *viper.Viper) error) error {
	stat := prome.NewStat("parseLocalConfig")
	defer stat.End()
	runtimeViper := viper.New()
	runtimeViper.SetConfigType("toml")
	fd, err := os.Open(filePath)
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("open  config error", zap.Error(err))
		return err
	}

	defer fd.Close()
	runtimeViper.ReadConfig(fd)

	if parseFunc != nil {
		err := parseFunc(runtimeViper)
		if err != nil {
			stat.MarkErr()
			zlog.LOG.Error("viper unmarshal PoolConfigure config error", zap.Error(err))
			return err
		}
	}

	return nil

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
