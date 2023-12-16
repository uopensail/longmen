package services

import (
	"context"
	"errors"

	longmenapi "github.com/uopensail/longmen/api"
	"github.com/uopensail/longmen/mgr"

	"github.com/gin-gonic/gin"
	"github.com/go-kratos/kratos/v2/registry"
	"github.com/uopensail/longmen/config"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/utils"
	"github.com/uopensail/ulib/zlog"
	etcdclient "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type Services struct {
	longmenapi.UnimplementedRankServer
	etcdCli *etcdclient.Client

	instance registry.ServiceInstance
}

func NewServices() *Services {
	srv := Services{}

	return &srv
}
func (srv *Services) Init(configFolder string, etcdName string, etcdCli *etcdclient.Client, reg utils.Register) {
	srv.etcdCli = etcdCli
	jobUtil := utils.NewMetuxJobUtil(etcdName, reg, etcdCli, 10, -1)
	mgr.MgrIns.Init(config.AppConfigInstance.EnvConfig, jobUtil)

}
func (srv *Services) RegisterGrpc(grpcS *grpc.Server) {
	longmenapi.RegisterRankServer(grpcS, srv)

}

func (srv *Services) RegisterGinRouter(ginEngine *gin.Engine) {
	apiV1 := ginEngine.Group("api/v1")
	{
		apiV1.POST("/rank", srv.RankHandler)
	}

}

func (srv *Services) RankHandler(c *gin.Context) {
	stat := prome.NewStat("App.RankHandler")
	defer stat.End()

	request := &longmenapi.Request{}
	if err := c.Bind(request); err != nil {
		zlog.LOG.Error("request bind error: ", zap.Error(err))
		return
	}
	resp, err := srv.Rank(context.Background(), request)
	if err != nil {
		zlog.LOG.Error("rank error: ", zap.Error(err))
		c.JSON(404, err.Error())
		return
	}
	c.JSON(200, resp)
	return
}

func (srv *Services) Rank(ctx context.Context, request *longmenapi.Request) (*longmenapi.Response, error) {
	if len(request.Records) <= 0 {
		return nil, errors.New("input empty")
	}
	itemIds := make([]string, len(request.Records))
	for i := 0; i < len(request.Records); i++ {
		itemIds[i] = request.Records[i].Id
	}
	scores, versions, err := mgr.MgrIns.Rank(request.UserFeatures, itemIds)
	resp := &longmenapi.Response{
		UserId:  request.UserId,
		Records: request.Records,
		Extras:  versions,
	}
	for i := 0; i < len(request.Records); i++ {
		resp.Records[i].Score = scores[i]
	}
	return resp, err
}

func (srv *Services) Check(context.Context, *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	return &grpc_health_v1.HealthCheckResponse{Status: grpc_health_v1.HealthCheckResponse_SERVING}, nil
}

func (srv *Services) Watch(*grpc_health_v1.HealthCheckRequest, grpc_health_v1.Health_WatchServer) error {
	return nil
}

func (srv *Services) Close() {

}
