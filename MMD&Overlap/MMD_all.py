import sys
sys.path.append("../evaluation")
from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict
import numpy as np
import datasets as da
from datasets import disable_caching
disable_caching()
from datasets import load_metric,load_dataset,Value
from transformers import AutoTokenizer,DataCollatorWithPadding
from datasets import concatenate_datasets
from transformers import AutoModel
import torch
import time, os
import json
from tqdm import *
from math import ceil
import pandas as pd
import transformers
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader




import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)


    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    loss = XX + YY - XY - YX
    return loss

from ranking import ood_tasks
from ranking import task_names_ID as glue_task_names

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

def preprocess_data_in_domain(args):
    data = load_dataset("./datasets/glue", args["task_name"], cache_dir= "./huggingface/datasets")
    # print(data)
    if "idx" in data["train"].features.keys():
        data = data.remove_columns("idx")
    if args["task_name"] == "stsb":
        data = data.map(lambda examples: {"label": round(examples["label"]/1.25)})
    if args["task_name"] == "qqp":
        data = data.rename_column("question1", "sentence1")
        data = data.rename_column("question2", "sentence2")
    if args["task_name"] == "mnli" or args["task_name"] == "mnli_matched" or args["task_name"] == "mnli_mismatched":
        data = data.rename_column("premise", "sentence1")
        data = data.rename_column("hypothesis", "sentence2")
    ########original#############
    if args["task_name"] == "qnli":
        data = data.rename_column("question", "sentence1")
        data = data.rename_column("sentence", "sentence2")
    return data["train"]

from utils_lp import preprocess_data as preprocess_data_ood


class glue_dataset(Dataset):
    def __init__(self, split, dataframe, tokenizer, args, seed, num):
        self.data = dataframe
        # self.tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
        self.max_len = args["max_len"]
        # print(self.max_len
        self.data = self.data.shuffle(seed=seed).select(range(num))
        self.num = num

        if args["type"] == "pair":
            # print("88888888888888888888888888888888888888888")
            # print(self.data)
            self.data = self.data.map(lambda examples: tokenizer(examples['sentence1'], examples['sentence2'],
                                                                 truncation=True, add_special_tokens=True,
                                                                 max_length=self.max_len, pad_to_max_length=True,
                                                                 # padding="max_length",
                                                                 return_token_type_ids=True, ),
                                      batched=True,
                                      load_from_cache_file=False)
            self.data = self.data.remove_columns(["sentence1", "sentence2"])
        elif args["type"] == "single":
            self.data = self.data.map(lambda examples: tokenizer(examples["sentence"],
                                                                 truncation=True, add_special_tokens=True,
                                                                 max_length=self.max_len, pad_to_max_length=True,
                                                                 #padding="max_length",
                                                                 return_token_type_ids=True, ),
                                      batched=True,
                                      load_from_cache_file=False)
            self.data = self.data.remove_columns("sentence")

    def __getitem__(self, index):
        inputs = self.data[index]
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        targets = inputs['label']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(int(targets), dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

class Classifier(torch.nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.l1 = AutoModel.from_pretrained("/yanglinyi/Models/"+os.path.basename(args["checkpoint"]))
        if "gpt2" in args["checkpoint"]:
            self.l1.config.pad_token_id = self.l1.config.eos_token_id  # for gpt2 only

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]  # choose the first embedding
        # print(pooler.shape)

        return pooler


def Evaluation(args, model, training_loader_0, training_loader_1):
    model.eval()

    output_0 = []
    output_1 = []

    with torch.no_grad():
        for _, data in enumerate(training_loader_0, 0):
            torch.cuda.empty_cache()
            ids = data['ids'].to(args["device"][0], dtype=torch.long)
            mask = data['mask'].to(args["device"][0], dtype=torch.long)
            # targets = data['targets'].to(args["device"][0], dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            output_0.append(outputs.to(args["device"][0]))

        for _, data in enumerate(training_loader_1, 0):
            torch.cuda.empty_cache()
            ids = data['ids'].to(args["device"][0], dtype=torch.long)
            mask = data['mask'].to(args["device"][0], dtype=torch.long)
            # targets = data['targets'].to(args["device"][0], dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            output_1.append(outputs.to(args["device"][0]))

        del model
        torch.cuda.empty_cache()
        # input("enter to continue")
        output_0 = torch.cat(output_0,dim=0)
        # print(output_0.shape)
        output_1 = torch.cat(output_1,dim=0)

        MMD_reuslt = mmd_rbf(source=output_0, target=output_1)
        print("MMD_result : ")
        print(MMD_reuslt)

    return MMD_reuslt


def fintune(args):
    time_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
    if "gpt2" in args["checkpoint"]:
        tokenizer.pad_token = tokenizer.eos_token  # for gpt2 only
    if args["mode"]=="id_vs_id" :
        args["task_name"] = args["dataset_0"]
        data_0 = preprocess_data_in_domain(args)
        args["task_name"] = args["dataset_1"]
        data_1 = preprocess_data_in_domain(args)
    # len_0 = len(data_0)
    elif args["mode"] == "id_vs_ood":
        args["task_name"] = args["dataset_0"]
        data_0 = preprocess_data_in_domain(args)
        args["task_name"] = args["dataset_1"]
        args["ID_name"] = "NULL"
        data_1 = preprocess_data_ood(args)
    elif args["mode"] == "ood_vs_ood":
        args["ID_name"] = "NULL"
        args["task_name"] = args["dataset_0"]
        data_0 = preprocess_data_ood(args)
        args["task_name"] = args["dataset_1"]
        data_1 = preprocess_data_ood(args)
    else:
        raise ValueError("mode not specified")
    # len_1 = len(data_1)

    args["task_name"] = ""

    training_set_0 = glue_dataset("train", data_0, tokenizer, args, seed=args["seed_0"], num=args["sample_len"])
    training_set_1 = glue_dataset("train", data_1, tokenizer, args, seed=args["seed_1"], num=args["sample_len"])

    train_params = {'batch_size': args["per_device_train_batch_size"]*len(args["device"]),
                    'shuffle': False,
                    'num_workers': 0
                    }
    training_loader_0 = DataLoader(training_set_0, **train_params)
    training_loader_1 = DataLoader(training_set_1, **train_params)

    model = Classifier(args)
    model = torch.nn.DataParallel(model, device_ids=args["device"])
    model = model.cuda(device=args["device"][0])

    return Evaluation(args, model, training_loader_0, training_loader_1), args["sample_len"]






def get_min_length(ID_task):
    args["task_name"] = ID_task
    len_ID = len(preprocess_data_in_domain(args))
    ood_len_list = []
    ood_len_list.append(len_ID)
    for ood_task in ood_tasks[ID_task]:
        args["task_name"] = ood_task
        ood_len_list.append( len( preprocess_data_ood(args) ) )
    args["task_name"] = ""
    return min(ood_len_list)

if __name__ == '__main__':
    # fintune(args)
    args = {

        "type": "",
        "checkpoint": "/yanglinyi/Models/roberta-base",
        "per_device_train_batch_size": 600,
        "max_len": 512,
        "device": [0, ],
        "sample_times": 20,

        "dataset_0": "",
        "seed_0": 0,
        "num": 1500,

        "dataset_1": "",
        "seed_1": 0,

        "common_len": "",
        "sample_len": "",

        "output_dir": "./MMD/all_new_1/"
        # "num_1":1800,

    }
    from utils_gluex import mkdir

    mkdir(args["output_dir"])
    for ID_task in glue_task_names:
        args["mode"] = "id_vs_id"
        args["type"] = types[ID_task]
        args["common_len"] = get_min_length(ID_task)
        args["sample_len"] = min(int(0.75*args["common_len"]), args["num"])
        args["dataset_0"] = ID_task
        args["dataset_1"] = ID_task
        result = []
        times = 0
        num = 0
        for i in range(args["sample_times"]):
            args["seed_1"] = i
            args["seed_0"] = i + args["sample_times"]
            evaluation, num = fintune(args)
            result.append(evaluation.item())
            times += 1

        mkdir(os.path.join(args["output_dir"], ID_task))
        output_result_path = os.path.join(args["output_dir"], ID_task, "MMD_between_{0}_and_{1}_sample_num_{2}.json".format(args["dataset_0"],args["dataset_1"], str(num)))

        with open(output_result_path, "w") as f:
            final_result = {
                "MMD_for_each": result,
                "avearge_MMD": sum(result) / len(result),
                "sample_times": times,
                "num": num,
                "args": args,
            }
            f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')
        print(result)





        #############################################################
        args["mode"] = "id_vs_ood"
        for OOD_task in ood_tasks[ID_task]:
            args["dataset_0"] = ID_task
            args["dataset_1"] = OOD_task
            result = []
            times = 0
            num = 0
            for i in range(args["sample_times"]):
                args["seed_1"] = i
                args["seed_0"] = i + args["sample_times"]
                evaluation, num = fintune(args)
                result.append(evaluation.item())
                times += 1

            mkdir(os.path.join(args["output_dir"], ID_task))
            output_result_path = os.path.join(args["output_dir"], ID_task,
                                              "MMD_between_{0}_and_{1}_sample_num_{2}.json".format(args["dataset_0"],
                                                                                                   args["dataset_1"],
                                                                                                   str(num)
                                                                                                   ),
                                              )

            with open(output_result_path, "w") as f:
                final_result = {
                    "MMD_for_each": result,
                    "avearge_MMD": sum(result) / len(result),
                    "sample_times": times,
                    "num": num,
                    "args": args,
                }
                f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')
            print(result)
        ###############################################新增#####################################################
        args["mode"] = "ood_vs_ood"
        for idx0, OOD_task_0 in enumerate(ood_tasks[ID_task]):
            for idx1, OOD_task_1 in enumerate(ood_tasks[ID_task][idx0:]):
                args["dataset_0"] = OOD_task_0
                args["dataset_1"] = OOD_task_1
                result = []
                times = 0
                num = 0
                for i in range(args["sample_times"]):
                    args["seed_1"] = i
                    args["seed_0"] = i + args["sample_times"]
                    evaluation, num = fintune(args)
                    result.append(evaluation.item())
                    times += 1

                mkdir(os.path.join(args["output_dir"], ID_task))
                output_result_path = os.path.join(args["output_dir"], ID_task,
                                                  "MMD_between_{0}_and_{1}_sample_num_{2}.json".format(args["dataset_0"],
                                                                                                       args["dataset_1"],str(num))
                                                  )

                with open(output_result_path, "w") as f:
                    final_result = {
                        "MMD_for_each": result,
                        "avearge_MMD": sum(result) / len(result),
                        "sample_times": times,
                        "num": num,
                        "args": args,
                    }
                    f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')
                print(result)