#!/bin/bash
# ================================================
# Script: run_train.sh
# Function: Run train.py in background with logging
# ================================================


echo ">>> Starting train.py ..."

# 日志文件带时间戳命名
log_file="train_$(date +%Y%m%d_%H%M%S).log"

# 后台运行 train.py，标准输出和错误输出都写入日志
nohup python -u train.py > "$log_file" 2>&1 &

# 获取后台进程 PID
pid=$!

echo ">>> train.py is now running in background"
echo ">>> Log file: $log_file"
echo ">>> PID: $pid"
echo ">>> To view log: tail -f $log_file"
echo ">>> To stop: kill $pid"