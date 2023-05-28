from utils_lp import parse_args
import os
import numpy as np
import json
import random,torch
from datasets import disable_caching
from eval_OOD_bert import predict_by_best_model
from eval_OOD_t5 import predict_by_best_model as predict_by_best_model_seq2seq
from utils_lp import get_ood_model_performance_path, get_ood_log_path, get_each_ood_task_log_path, reproduce
from datasets import set_caching_enabled


os.environ['TRANSFORMERS_CACHE'] = './huggingface'
args = parse_args()
print(args)

device_ids = args["device"]
random_seed = args["random_seed"]
reproduce(random_seed)
disable_caching()
set_caching_enabled(False)
ID_names =  args["TASK_NAMES"]
ood_tasks_specified = json.loads(args["OOD_TASKS_SPECIFIED"])
checkpoints = args["CHECKPOINTS_LIST"]

batch_size_A100 = {
    "../Models/roberta-base": 1000,
    "../Models/roberta-large": 1000,
    "../Models/xlnet-large-cased": 1000,
    "../Models/xlnet-base-cased": 1000,
    "../Models/distilbert-base-uncased": 1000,
    "../Models/albert-base-v2": 1000,
    "../Models/bert-base-uncased": 1000,
    "../Models/bert-large-uncased": 1000,
    "../Models/electra-small-discriminator": 1000,
    "../Models/electra-base-discriminator": 1000,
    "../Models/electra-large-discriminator": 1000,

    "../Models/gpt2": 500,
    "../Models/gpt2-medium": 500,
    "../Models/gpt2-large": 200,
    "../Models/bart-base": 500,
    "../Models/bart-large": 500,

    "../Models/t5-small": 850,
    "../Models/t5-base": 620,
    "../Models/t5-large": 450,
}
batch_size_V100={
    "../Models/roberta-base": 80,
    "../Models/roberta-large": 50,
    "../Models/xlnet-large-cased": 50,
    "../Models/xlnet-base-cased": 80,
    "../Models/distilbert-base-uncased": 500,
    "../Models/albert-base-v2": 500,
    "../Models/bert-base-uncased": 50,
    "../Models/bert-large-uncased": 30,
    "../Models/electra-small-discriminator": 500,
    "../Models/electra-base-discriminator": 200,
    "../Models/electra-large-discriminator": 10,

    "../Models/gpt2": 80,
    "../Models/gpt2-medium": 60,
    "../Models/gpt2-large": 30,
    "../Models/bart-base": 50,
    "../Models/bart-large": 30,

    "../Models/t5-small": 200,
    "../Models/t5-base": 100,
    "../Models/t5-large": 30,
}
if args["device_type"] == "A100":
    batch_size = batch_size_A100
elif args["device_type"] == "V100":
    batch_size = batch_size_V100
else:
    raise ValueError("device type ont supported!")

# checkpoints=["bert-base-uncased"]
#     "sst2": "./sst2"
# }
labels = {
    "sst2": 2,
    "cola": 2,
    "mrpc": 2,
    "stsb": 5,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2
}
eval_metrics={
    "sst2": "accuracy",
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "stsb": "pearson",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
}
greater_is_better={
    "sst2": True,
    "cola": True,
    "mrpc": True,
    "stsb": True,
    "qqp": True,
    "mnli": True,
    "qnli": True,
    "rte": True,
    "wnli": True,

}
types={
    "sst2": "single",
    "cola": "single",
    "mrpc": "pair",
    "stsb": "pair",
    "qqp": "pair",
    "mnli": "pair",
    "qnli": "pair",
    "rte": "pair",
    "wnli": "pair",
}

import time

from utils_gluex import get_best_model_dict
def get_best_model_dirs(args):
    best_model_dirs={}
    for checkpoint in checkpoints:
        checkpoint = os.path.basename(checkpoint)
        best_model_dirs[checkpoint] = get_best_model_dict(os.path.join(args["model_performance_dir"], checkpoint, checkpoint+args["model_performance_suffix"]))
    return best_model_dirs


best_model_dirs = get_best_model_dirs(args)
print(best_model_dirs)

results={}

for checkpoint in checkpoints:
    args["checkpoint"] = checkpoint
    args["per_device_eval_batch_size"] = batch_size[checkpoint]
    results[checkpoint] = {}
    for ID_name in ID_names:
        if ID_name not in best_model_dirs[os.path.basename(checkpoint)].keys():
            raise ValueError("The {0} key is not in {1}".format(ID_name, best_model_dirs[os.path.basename(checkpoint)]))
            continue
        args["best_model_dir"] = best_model_dirs[os.path.basename(checkpoint)][ID_name]
        args["ID_name"] = ID_name
        args["type"] = types[ID_name]
        args["num_labels"] = labels[ID_name]
        results[checkpoint]["ood_for_"+ID_name] = {}
        results[checkpoint]["ood_for_"+ID_name]["ood_average_evaluation"] = 0
        total_length = 0
        total_cases = 0
        for ood_task in ood_tasks_specified[ID_name]:
            time_start = time.time()
            print("ID_name:{0}, OOD_task:{1}, Model_name:{2}".format(ID_name, ood_task, checkpoint))
            args["task_name"] = ood_task
            if args["seq2seq"]:
                ood_evaluation_result, length = predict_by_best_model_seq2seq(args)
            else:
                ood_evaluation_result, length = predict_by_best_model(args)
            total_cases+=length
            time_end = time.time()
            results[checkpoint]["ood_for_"+ID_name][ood_task] = ood_evaluation_result
            results[checkpoint]["ood_for_"+ID_name][ood_task]["length"] = length
            results[checkpoint]["ood_for_"+ID_name][ood_task]["time_consumed"] = time_end - time_start

            for k,v in ood_evaluation_result.items():
                results[checkpoint]["ood_for_" + ID_name]["ood_average_evaluation"] += length * v
                total_length+=length
            each_task_save_path = get_each_ood_task_log_path(args)
            with open(each_task_save_path, "w") as f:
                f.write(json.dumps(results[checkpoint]["ood_for_" + ID_name][ood_task], ensure_ascii=False, indent=4,
                                   separators=(',', ':')) + '\n')


        results[checkpoint]["ood_for_" + ID_name]["ood_average_evaluation"] /= total_length
        results[checkpoint]["ood_for_" + ID_name]["total_cases"] = total_cases

