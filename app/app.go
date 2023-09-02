package app

import (
	"context"
	"longmen/api"
	"longmen/model"

	"github.com/gin-gonic/gin"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

var __GITHASH__ = ""

type App struct {
	api.UnimplementedRankServer
}

func NewApp() *App {
	model.Init()
	app := &App{}
	return app
}

func (app *App) Close() {
	model.Close()
}

func (app *App) GRPCAPIRegister(s *grpc.Server) {
	api.RegisterRankServer(s, app)
}

func (app *App) GinAPIRegister(e *gin.Engine) {
	e.POST("/rank", app.RankHandler)
	e.GET("/version", app.VersionHandler)
	e.GET("/ping", app.PingHandler)
}

func (app *App) RankHandler(c *gin.Context) {
	stat := prome.NewStat("App.RankHandler")
	defer stat.End()

	request := &api.Request{}
	if err := c.Bind(request); err != nil {
		zlog.LOG.Error("request bind error: ", zap.Error(err))
		return
	}
	resp, err := app.Rank(context.Background(), request)
	if err != nil {
		zlog.LOG.Error("rank error: ", zap.Error(err))
		c.JSON(404, err.Error())
		return
	}
	c.JSON(200, resp)
	return
}

func (app *App) Rank(ctx context.Context, request *api.Request) (*api.Response, error) {
	resp, err := model.Rank(request)
	return resp, err
}

func (app *App) PingHandler(c *gin.Context) {
	c.String(200, "PONG")
}

func (app *App) VersionHandler(c *gin.Context) {
	c.String(200, __GITHASH__)
}

func (app *App) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

func (app *App) Watch(req *grpc_health_v1.HealthCheckRequest, server grpc_health_v1.Health_WatchServer) error {
	server.Send(&grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	})
	return nil
}
