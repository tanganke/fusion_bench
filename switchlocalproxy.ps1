# 检查当前仓库是否已设置 http.proxy（仅 local 配置）
$currentProxy = git config --local --get http.proxy

if ($currentProxy) {
    # 如果已设置本地代理，则关闭本地代理
    Write-Output "检测到本地代理已开启，现在关闭本地代理..."
    git config --local --unset http.proxy
    git config --local --unset https.proxy
    Write-Output "本地代理已关闭。"
} else {
    # 如果未设置本地代理，则开启本地代理
    Write-Output "本地代理未开启，现在开启本地代理..."
    git config --local http.proxy http://127.0.0.1:7890
    git config --local https.proxy http://127.0.0.1:7890
    Write-Output "本地代理已开启。"
}
