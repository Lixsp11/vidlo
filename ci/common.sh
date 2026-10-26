# ci/ 下脚本共用的函数，用 source 引入

download() {  # url 文件 sha256；curl 的 --retry 不重试 TLS 握手失败（退出码 35）这类连接错误，这里自己重试
  for attempt in 1 2 3 4 5; do
    curl -fsSL -o "$2" "$1" && break
    [ "$attempt" = 5 ] && return 1
    sleep $((attempt * 10))
  done
  echo "$3  $2" | sha256sum -c -
}
