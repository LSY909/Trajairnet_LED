
#!/bin/bash
# ================================================
# Script: run_test.sh
# Function: Run test.py in background with logging
# ================================================


echo ">>> Starting test.py ..."

# 日志文件带时间戳命名
log_file="test_$(date +%Y%m%d_%H%M%S).log"

# 后台运行 test.py，标准输出和错误输出都写入日志
nohup python test.py > "$log_file" 2>&1 &

# 获取后台进程 PID
pid=$!

echo ">>> test.py is now running in background"
echo ">>> Log file: $log_file"
echo ">>> PID: $pid"
echo ">>> To view log: tail -f $log_file"
echo ">>> To stop: kill $pid"
