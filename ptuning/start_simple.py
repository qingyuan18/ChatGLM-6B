import os
import json
import socket

if __name__ == "__main__":
    
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))  

    #pass env parameters
    os.environ['MODEL_S3_PATH'] =  str( os.environ['MODEL_S3_PATH'])                        
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = str( os.environ['PYTORCH_CUDA_ALLOC_CONF'])
    os.environ['LD_LIBRARY_PATH']  = str(os.environ['LD_LIBRARY_PATH'])
    os.environ['TRAIN_DATASET'] = str(os.environ['TRAIN_DATASET'])
    os.environ['TEST_DATASET'] = str(os.environ['TEST_DATASET'])
    os.environ['PROMPT_COLUMN'] = str(os.environ['PROMPT_COLUMN'])
    os.environ['RESPONSE_COLUMN'] = str(os.environ['RESPONSE_COLUMN'])
    os.environ['MODEL_NAME_OR_PATH'] = str(os.environ['MODEL_NAME_OR_PATH'])
    os.environ['OUTPUT_DIR'] = str(os.environ['OUTPUT_DIR'])
    os.environ['MODEL_OUTPUT_S3_PATH'] = str(os.environ['MODEL_OUTPUT_S3_PATH'])

    
    #invoke the torch launcher shell script.
    #Note: we will use the pytorch launcher to launch deepspeed for multi-nodes training.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash train-simple.sh")    