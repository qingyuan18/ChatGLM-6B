# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import uuid
import io
import sys

import traceback

from PIL import Image

import requests
import boto3
import sagemaker
import torch


from torch import autocast
#from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_p=0.45, temperature=0.7,model=None):
    text = preprocess(text)
    response, history = model.chat(tokenizer, text, history=[],max_length=100, top_p=top_p,
                                   temperature=temperature)

    return postprocess(response)


def model_fn(model_dir):
    print("=================model_fn_Start=================")
    model_s3_path = os.environ['MODEL_S3_PATH']
    print("=================model s3 path=================="+model_s3_path)
    os.system("cp ./code/s5cmd  /tmp/ && chmod +x /tmp/s5cmd")
    os.system("/tmp/s5cmd sync {0} {1}".format(model_s3_path+"*","/tmp/model/"))
    if os.environ["MODEL_TYPE"] == "ptuning":
        config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join("/tmp/model/", "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    elif os.environ["MODEL_TYPE"] == "full turning":
        print("====================load full turning=================")
        config = AutoConfig.from_pretrained("/tmp/model/", trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained("/tmp/model/", trust_remote_code=True)
    else:
        print("====================load normal ======================")
        config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    #model = model.to("cuda")
    model = model.quantize(4)
    model = model.half().cuda()
    model = model.eval()
    print("=================model_fn_End=================")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    # {
    # "ask": "写一个文章，题目是未来城市"
    # }
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    if 'ask' not in input_data:
        input_data['ask']="写一个文章，题目是未来城市"
    return input_data




def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    print("=================predict_fn=================")

    print('input_data: ', input_data)


    try:
        result=answer(input_data['ask'], model=model)
        print(f'====result {result}====')
        return result

    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception================={ex}")

    return 'Not found answer'


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print(content_type)
    return json.dumps(
        {
            'answer': prediction
        }
    )



