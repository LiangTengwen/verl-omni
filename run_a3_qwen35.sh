#!/bin/bash
set -ex

pip install mathruler

export TORCH_EXTENSIONS_DIR=/cache/.torch_extension
export NPUS_PER_NODE=16

MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_ADDR=$(ping "$MASTER_ADDR" -c 1 | sed '1{s/[^(]*(//;s/).*//;q}')
RAY_PORT=6925
RAY_DASHBOARD_PORT=8960

############################## COMMON FUN ##########################

rm_exist_dir(){
    if [ -d "$1" ] ; then
      rm -rf "$1"
    fi
}

print_stage(){
  echo "###################  $1  #############################"
}

print_stage "ENV PRINT"
echo "$@"

#############################  PRE CHECK ##########################
print_stage "PRE CHECK"
WORKER_RANK=${VC_TASK_INDEX:-0}
DEPLOY_LOG_PATH="/opt/huawei/schedule-train/log/$WORKER_RANK"
MASTER_INFO_FILE="/opt/huawei/schedule-train/log/0/master_info.txt"
WORKER_LOG_PATH="$DEPLOY_LOG_PATH/worker_log_$WORKER_RANK"

mkdir -p "$DEPLOY_LOG_PATH"

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export USE_OPTIMIZED_MODEL=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export NPU_GRAPH_CAPTURE_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NPUS_PER_NODE}
export VLLM_NO_SUBCLASS_INSPECT=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [ -f "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
fi

export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export PYTHONPATH=/opt/huawei/dataset/ccb_data/code/hufenyu/verl-omni/:/opt/huawei/dataset/ccb_data/code/package_install/verl:/opt/huawei/dataset/ccb_data/code/MiniCPM:$PYTHONPATH
ulimit -s unlimited

############################### START RAY CLUSTER ###############################
TOTAL_NODES=${VC_WORKER_NUM:-1}

if [ "$WORKER_RANK" -eq 0 ]; then
  print_stage "START RAY HEAD"
  
  ray start --head \
    --port="$RAY_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --resources="{\"NPU\": $NPUS_PER_NODE}"

  echo "${MASTER_ADDR}:${RAY_PORT}" > "$MASTER_INFO_FILE"
  
  print_stage "WAITING FOR ALL $TOTAL_NODES NODES TO JOIN RAY CLUSTER"

  MAX_RETRIES=60
  RETRY_INTERVAL=5

  for ((i=1; i<=MAX_RETRIES; i++)); do
    ACTIVE_NODES=$(ray status 2>/dev/null | awk '/^Active:/{flag=1; next} /^Pending:/{flag=0} flag && /^ +[0-9]+/ {count++} END {print count+0}')
    echo "[$i/$MAX_RETRIES] Current Active Nodes: $ACTIVE_NODES / Target Nodes: $TOTAL_NODES"
    
    if [ "$ACTIVE_NODES" -ge "$TOTAL_NODES" ]; then
      echo "All $TOTAL_NODES nodes have successfully joined the Ray cluster."
      break
    fi

    if [ "$i" -eq "$MAX_RETRIES" ]; then
      echo "Error: Timed out waiting for all nodes to join. Expected: $TOTAL_NODES, Found: $ACTIVE_NODES"
      exit 1
    fi

    sleep $RETRY_INTERVAL
  done
  cd /opt/huawei/dataset/ccb_data/code/hufenyu/verl-omni/examples/gspo_trainer/qwen35_moe
  bash run_qwen35_moe_thinker_gspo_npu.sh
  # sleep 86400

else
  print_stage "THIS IS WORKER (RANK: $WORKER_RANK)"
  
  for ((i=0; i<100; i++)); do
    if [ -f "$MASTER_INFO_FILE" ]; then
      echo "Master info file exists."
      break
    fi
    echo "Waiting for master to start... ($i/100)"
    sleep 5
  done

  if [ ! -f "$MASTER_INFO_FILE" ]; then
    echo "The master process is not ready after 500s."
    exit 1
  fi

  MASTER_INFO=$(head -n 1 "$MASTER_INFO_FILE")
  [ -z "$MASTER_INFO" ] && MASTER_INFO="${MASTER_ADDR}:${RAY_PORT}"

  ray start \
    --address="$MASTER_INFO" \
    --resources="{\"NPU\": $NPUS_PER_NODE}" \
    --block
fi