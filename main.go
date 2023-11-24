package main

import (
	"context"
	"flag"
	"fmt"

	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path"
	"syscall"
	"time"

	etcd "github.com/go-kratos/kratos/contrib/registry/etcd/v2"
	"github.com/go-kratos/kratos/v2"
	"github.com/go-kratos/kratos/v2/middleware/recovery"
	"github.com/go-kratos/kratos/v2/registry"
	kgrpc "github.com/go-kratos/kratos/v2/transport/grpc"
	khttp "github.com/go-kratos/kratos/v2/transport/http"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"go.uber.org/zap"

	"google.golang.org/grpc"

	"github.com/gin-gonic/gin"
	"github.com/uopensail/longmen/config"
	"github.com/uopensail/longmen/services"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	etcdclient "go.etcd.io/etcd/client/v3"
)

var __GITCOMMITINFO__ = ""

// PingPongHandler @Summary 获取标签列表
// @BasePath /
// @Produce  json
// @Success 200 {object} model.StatusResponse
// @Router /ping [get]
func PingPongHandler(gCtx *gin.Context) {
	pStat := prome.NewStat("PingPongHandler")
	defer pStat.End()

	gCtx.JSON(http.StatusOK, struct {
		Code int    `json:"code"`
		Msg  string `json:"msg"`
	}{
		Code: 0,
		Msg:  "PONG",
	})
	return
}

// @Summary 获取标签列表
// @BasePath /
// @Produce  json
// @Success 200 {object} model.StatusResponse
// @Router /git_hash [get]
func GitHashHandler(gCtx *gin.Context) {
	pStat := prome.NewStat("GitHashHandler")
	defer pStat.End()

	gCtx.String(http.StatusOK, "git_info:"+__GITCOMMITINFO__)
	return
}

type kratosAppRegister struct {
	registry.Registrar
	*kratos.App
}

func (reg *kratosAppRegister) buildInstance() *registry.ServiceInstance {
	instance := registry.ServiceInstance{}
	instance.ID = reg.App.ID()
	instance.Name = reg.App.Name()
	instance.Version = reg.App.Version()
	instance.Endpoints = reg.App.Endpoint()
	instance.Metadata = reg.App.Metadata()
	return &instance
}
func (reg *kratosAppRegister) Register(ctx context.Context) error {
	instance := reg.buildInstance()
	return reg.Registrar.Register(ctx, instance)
}

func (reg *kratosAppRegister) Deregister(ctx context.Context) error {
	instance := reg.buildInstance()
	return reg.Registrar.Deregister(ctx, instance)
}

func run(configFilePath string, logDir string) *services.Services {
	config.AppConfigInstance.Init(configFilePath)
	folder := path.Dir(configFilePath)
	zlog.InitLogger(config.AppConfigInstance.ProjectName, config.AppConfigInstance.Debug, logDir)

	var etcdCli *etcdclient.Client
	if len(config.AppConfigInstance.ServerConfig.Endpoints) > 0 {
		client, err := etcdclient.New(etcdclient.Config{
			Endpoints: config.AppConfigInstance.ServerConfig.Endpoints,
		})
		if err != nil {
			zlog.LOG.Fatal("etcd error", zap.Error(err))
		} else {
			etcdCli = client
		}
	}

	options := make([]kratos.Option, 0)
	var reg registry.Registrar

	if etcdCli != nil {
		reg = etcd.New(etcdCli)
		options = append(options, kratos.Registrar(reg))
	}
	serverName := config.AppConfigInstance.ServerConfig.Name
	services := services.NewServices()
	grpcSrv := newGRPC(services.RegisterGrpc)
	httpSrv := newHTTPServe(services.RegisterGinRouter)

	options = append(options, kratos.Name(serverName), kratos.Version(__GITCOMMITINFO__), kratos.Server(
		httpSrv,
		grpcSrv,
	))
	app := kratos.New(options...)
	appReg := kratosAppRegister{
		Registrar: reg,
		App:       app,
	}
	services.Init(folder, "microservices/"+serverName, etcdCli, &appReg)
	go func() {
		if err := app.Run(); err != nil {
			zlog.LOG.Fatal("run error", zap.Error(err))
		}
	}()

	return services
}

func newHTTPServe(registerFunc func(*gin.Engine)) *khttp.Server {
	ginEngine := gin.New()
	ginEngine.Use(gin.Recovery())

	url := ginSwagger.URL(fmt.Sprintf("swagger/doc.json")) // The url pointing to API definition
	ginEngine.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler, url))

	ginEngine.GET("/ping", PingPongHandler)
	ginEngine.GET("/git_hash", GitHashHandler)

	registerFunc(ginEngine)
	httpSrv := khttp.NewServer(khttp.Address(fmt.Sprintf(":%d", config.AppConfigInstance.ServerConfig.HttpServerConfig.HTTPPort)))
	httpSrv.HandlePrefix("/", ginEngine)
	return httpSrv
}

func newGRPC(registerFunc func(server *grpc.Server)) *kgrpc.Server {
	grpcSrv := kgrpc.NewServer(
		kgrpc.Address(fmt.Sprintf(":%d",
			config.AppConfigInstance.ServerConfig.GRPCPort)),
		kgrpc.Middleware(
			recovery.Recovery(),
		),
	)
	registerFunc(grpcSrv.Server)
	return grpcSrv
}

func runPProf(port int) {
	if port > 0 {
		go func() {
			fmt.Println(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", port), nil))
		}()
	}
}

func runProme(projectName string, port int) *prome.Exporter {
	//prome的打点
	promeExport := prome.NewExporter(projectName)
	go func() {
		err := promeExport.Start(port)
		if err != nil {
			panic(err)
		}
	}()

	return promeExport
}

// @title Swagger Example API
// @version 1.0
// @description This is a sample server Petstore server.
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.url http://www.swagger.io/support
// @contact.email support@swagger.io

// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html
// @BasePath /api/v1

// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name Authorization

func main() {
	configFilePath := flag.String("config", "conf/config.toml", "启动命令请设置配置文件目录")
	logDir := flag.String("log", "./logs", "启动命令请设置seelog.xml")
	flag.Parse()

	application := run(*configFilePath, *logDir)

	if len(config.AppConfigInstance.ProjectName) <= 0 {
		panic("config.ProjectName NULL")
	}

	runPProf(config.AppConfigInstance.PProfPort)

	//prome的打点
	promeExport := runProme(config.AppConfigInstance.ProjectName, config.AppConfigInstance.PromePort)
	signalChanel := make(chan os.Signal, 1)
	signal.Notify(signalChanel, syscall.SIGINT, syscall.SIGTERM)
	fmt.Println(time.Now().Format("2006-01-02 15:04:05"), " app running....")
	<-signalChanel
	application.Close()
	promeExport.Close()
	fmt.Println(time.Now().Format("2006-01-02 15:04:05"), " app exit....")
}
