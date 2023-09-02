package config

import (
	"os"
	"strings"

	"github.com/BurntSushi/toml"
	"github.com/uopensail/ulib/commonconfig"
)

type PoolConfigure struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Key     string `json:"key" toml:"key" yaml:"key"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type ModelConfigure struct {
	Path    string `json:"path" toml:"path" yaml:"path"`
	Pool    string `json:"pool" toml:"pool" yaml:"pool"`
	Kit     string `json:"kit" toml:"kit" yaml:"kit"`
	Version string `json:"version" toml:"version" yaml:"version"`
}

type AppConfig struct {
	commonconfig.ServerConfig `json:"server" toml:"server" yaml:"server"`
	Pools                     []string `json:"pools" toml:"pools" yaml:"pools"`
	Models                    []string `json:"models" toml:"models" yaml:"models"`
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

var AppConf AppConfig
