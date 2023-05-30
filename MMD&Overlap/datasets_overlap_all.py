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
import argparse

from MMD_all import ood_tasks
from MMD_all import glue_task_names
from MMD_all import types

from MMD_all import preprocess_data_ood, preprocess_data_in_domain

def get_data(args):
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

    return data_0, data_1




import string
punc = string.punctuation

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def overlap(dataset0, dataset1, args):

    word_set0 = set()
    word_set1 = set()
    for d0 in dataset0:
        if args["type"] == "single":
            sentence = d0["sentence"]
        elif args["type"] =='pair':
            sentence0 = d0["sentence1"]
            sentence1 = d0["sentence2"]
            sentence = sentence0 + sentence1
        else:
            raise ValueError("type not specified")
        words0 = word_tokenize(sentence)
        for word0 in words0:
            if word0 not in punc:
                word_set0.add(word0)

    for d1 in dataset1:
        if args["type"] == "single":
            sentence = d1["sentence"]
        elif args["type"] =='pair':
            sentence0 = d1["sentence1"]
            sentence1 = d1["sentence2"]
            sentence = sentence0 + sentence1
        else:
            raise ValueError("type not specified")
        words1 = word_tokenize(sentence)
        for word1 in words1:
            if word1 not in punc:
                word_set1.add(word1)

    intersect = word_set0.intersection(word_set1)

    if len(intersect) == 0:
        return 0

    precision = len(intersect) / len(word_set0)
    recall = len(intersect) / len(word_set1)

    f1 = 2*precision*recall / (precision + recall)
    return f1, precision, recall


def save_result(save_path, f1, precision, recall):
    with open(save_path, "w") as f:
        final_result = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        print(final_result)
        f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    args = {
        "output_dir": "./datasets_overlap_1/"
    }
    mkdir(args["output_dir"])
    parser = argparse.ArgumentParser("Demo of argparse")
    ARGS = parser.parse_args()

    for ID_task in glue_task_names:
        args['type'] = types[ID_task]
        args["mode"] = "id_vs_id"
        args["dataset_0"] = ID_task
        args["dataset_1"] = ID_task
        f1 = 1.0
        precision = 1.0
        recall = 1.0
        save_dir = os.path.join(args["output_dir"], ID_task)
        mkdir(save_dir)
        save_path = os.path.join(save_dir, "overlap_between_{0}_and_{1}.json".format(ID_task, ID_task))
        save_result(save_path, f1, precision, recall)

        for ood_task in ood_tasks[ID_task]:
            args["mode"] = "id_vs_ood"
            args["dataset_0"] = ID_task
            args["dataset_1"] = ood_task
            data0, data1 = get_data(args)
            f1, precision, recall = overlap(data0, data1, args)
            save_path = os.path.join(save_dir, "overlap_between_{0}_and_{1}.json".format(ID_task, ood_task))
            save_result(save_path, f1, precision, recall)



        for idx0, OOD_task_0 in enumerate(ood_tasks[ID_task]):
            for idx1, OOD_task_1 in enumerate(ood_tasks[ID_task][idx0:]):
                args["mode"] = "ood_vs_ood"
                args["dataset_0"] = OOD_task_0
                args["dataset_1"] = OOD_task_1
                if args["dataset_0"] == args["dataset_1"]:
                    f1 = 1.0
                    precision = 1.0
                    recall = 1.0
                else:
                    data0, data1 = get_data(args)
                    f1, precision, recall = overlap(data0, data1, args)
                save_path = os.path.join(save_dir, "overlap_between_{0}_and_{1}.json".format(OOD_task_0, OOD_task_1))
                save_result(save_path, f1, precision, recall)


