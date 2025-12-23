#!/bin/bash
#!/bin/bash

###---------------------使用离线预计算的 route_priors
echo "Running train.py in background (dataset=7days1)..."
log_file="train_7days1_$(date +%Y%m%d_%H%M%S).log"
nohup /opt/conda/envs/env_traj/bin/python train.py \
    --dataset_name 7days1 \
    --route_priors_train ./dataset/route_priors_7days1_train.pt \
    --route_priors_test  ./dataset/route_priors_7days1_test.pt \
    > "$log_file" 2>&1 &

echo "Log: $log_file"
echo "PID: $!"