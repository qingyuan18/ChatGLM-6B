import os
import json
import socket

if __name__ == "__main__":

    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))

    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)

    hosts = json.loads(os.environ['SM_HOSTS'])
    num_gpus = int(os.environ['NUM_GPUS'])
    with open('./hosts', 'w') as f:
        for host in hosts:
            line = f"{host} {num_gpus}\n"
            f.write(line)

    #pass env parameters
    os.environ['MODEL_S3_PATH'] =  str( os.environ['MODEL_S3_PATH'])
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = str( os.environ['PYTORCH_CUDA_ALLOC_CONF'])
    os.environ['LD_LIBRARY_PATH']  = str(os.environ['LD_LIBRARY_PATH'])
    os.environ['NUM_GPUS']  = int(os.environ['NUM_GPUS'])
    os.environ['TRAIN_DATASET'] = str(os.environ['TRAIN_DATASET'])
    os.environ['TEST_DATASET'] = str(os.environ['TEST_DATASET'])
    os.environ['PROMPT_COLUMN'] = str(os.environ['PROMPT_COLUMN'])
    os.environ['RESPONSE_COLUMN'] = str(os.environ['RESPONSE_COLUMN'])
    os.environ['MODEL_NAME_OR_PATH'] = str(os.environ['MODEL_NAME_OR_PATH'])
    os.environ['OUTPUT_DIR'] = str(os.environ['OUTPUT_DIR'])
    os.environ['MODEL_OUTPUT_S3_PATH'] = str(os.environ['MODEL_OUTPUT_S3_PATH'])

    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    #invoke the torch launcher shell script.
    #Note: we will use the pytorch launcher to launch deepspeed for multi-nodes training.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash ds_train_finetune.sh")