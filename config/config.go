package config

import (
	"io/ioutil"
	"strings"

	"github.com/BurntSushi/toml"
	"github.com/uopensail/ulib/commonconfig"
	"gopkg.in/yaml.v2"
)

type EnvConfig struct {
	Finder  commonconfig.FinderConfig `json:"finder" toml:"finder"`
	WorkDir string                    `json:"work_dir" toml:"work_dir"`
}
type ModelConfig struct {
	Name    string `json:"name" toml:"name" yaml:"name"`
	Path    string `json:"path" toml:"path" yaml:"path" mapstructure:"path"`
	Version string `json:"version" toml:"version" yaml:"version"`
}
type PoolConfig struct {
	Name    string `json:"name" toml:"name" yaml:"name"`
	Path    string `json:"path" toml:"path" yaml:"path"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	EnvConfig                 `json:"env" toml:"env" yaml:"env"`

	BigPoolVersionFile   string `json:"big_pool" toml:"big_pool" yaml:"big_pool"`
	SmallPoolVersionFile string `json:"small_pool" toml:"small_pool" yaml:"small_pool"`
	ModelVersionFile     string `json:"model" toml:"model" yaml:"model"`
}

func (conf *AppConfig) ReadBigPoolVersion() (PoolConfig, error) {
	cfg := struct {
		PoolConfig `json:"pool" toml:"pool" yaml:"pool"`
	}{}
	fileData, err := ioutil.ReadFile(conf.BigPoolVersionFile)
	if err != nil {
		return cfg.PoolConfig, err
	}

	if _, err := toml.Decode(string(fileData), &cfg); err != nil {
		return cfg.PoolConfig, err
	}
	return cfg.PoolConfig, nil
}

func (conf *AppConfig) ReadSmallPoolVersion() (PoolConfig, error) {
	cfg := struct {
		PoolConfig `json:"pool" toml:"pool" yaml:"pool"`
	}{}
	fileData, err := ioutil.ReadFile(conf.SmallPoolVersionFile)
	if err != nil {
		return cfg.PoolConfig, err
	}

	if _, err := toml.Decode(string(fileData), &cfg); err != nil {
		return cfg.PoolConfig, err
	}
	return cfg.PoolConfig, nil
}

func (conf *AppConfig) ReadModelVersion() (ModelConfig, error) {
	cfg := struct {
		ModelConfig `json:"model" toml:"model" yaml:"model"`
	}{}
	fileData, err := ioutil.ReadFile(conf.ModelVersionFile)
	if err != nil {
		return cfg.ModelConfig, err
	}

	if _, err := toml.Decode(string(fileData), &cfg); err != nil {
		return cfg.ModelConfig, err
	}
	return cfg.ModelConfig, nil
}

func (conf *AppConfig) Init(filePath string) {
	fileData, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	if strings.HasSuffix(filePath, ".toml") {
		if _, err := toml.Decode(string(fileData), conf); err != nil {
			panic(err)
		}
	} else {
		if err := yaml.Unmarshal(fileData, conf); err != nil {
			panic(err)
		}

	}

}

var AppConfigIns AppConfig
