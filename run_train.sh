
##!/bin/bash
echo "Running train.py in background..."
log_file="train_$(date +%Y%m%d_%H%M%S).log"

# 后台运行 train.py
nohup python train.py > "$log_file" 2>&1 &

pid=$!   # 记录后台进程 PID

echo ">>> train.py is now running in background"
echo ">>> Log file: $log_file"
echo ">>> PID: $pid"
echo ">>> To view log: tail -f $log_file"
echo ">>> To stop: kill $pid"
