#!/bin/bash
NUM_PROC=$1
shift
torchrun --nproc_per_node=$NUM_PROC train.py "$@"

#!/bin/bash
#NUM_PROC=$1
#shift

# Set the master address and port
#MASTER_ADDR=${MASTER_ADDR:-"192.168.5.55"}  # Replace "localhost" with the IP of the master node
#MASTER_PORT=${MASTER_PORT:-29500}

# Set the number of nodes and the rank of this node
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}

# Run the training script with torchrun
#torchrun --nproc_per_node=$NUM_PROC \
#         --nnodes=$NNODES \
#         --node_rank=$NODE_RANK \
#         --master_addr=$MASTER_ADDR \
#         --master_port=$MASTER_PORT \
#         train.py "$@"
