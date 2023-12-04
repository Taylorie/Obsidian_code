## docker

## docker 镜像全部启动命令 

```shell
docker start $(docker ps -a | awk '{ print $1}' | tail -n +2)

```

