#coding=utf-8
from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict, calcuate_accu_for_str_list, map_seq_to_value, mode
import numpy as np
import datasets as da
import json

import pandas as pd
from datasets import disable_caching
disable_caching()
from datasets import load_metric,load_dataset,Value
from datasets import concatenate_datasets

from transformers import AutoTokenizer,DataCollatorWithPadding
from transformers import AutoModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time, os
os.environ['TRANSFORMERS_CACHE'] = './huggingface'
import json
from tqdm import *
from math import ceil
import pandas as pd


import transformers,xlrd
from torch.utils.data import Dataset, DataLoader

args = {
    "datasets_dir":"./sampled_ood_data",
    "human_annotated_dir":"./gluex_human_annotated",
    "table_save_dir":"./sampled_ood_data_tables",
    "save_dir":"./GLUE_human_annotated_results"
}
mkdir(args["save_dir"])
data = load_dataset("./datasets/glue", 'stsb', cache_dir="./huggingface/datasets")
print(min(data["train"]["label"]))
print(max(data["train"]["label"]))

# input()
def read_annotation(file_path, sheet, original_table_name, ID_name):
    original_table_path = os.path.join(args["table_save_dir"], original_table_name)
    GT_value = pd.read_csv(original_table_path)
    if original_table_name == "sampled_qnli_ood.csv":
        GT_value = [int(each) for each in GT_value["GroundTruth_label_converted"]]
    else:
        GT_value = [int(each) for each in GT_value["GroundTruth_original_label"]]

    wb = xlrd.open_workbook(file_path)
    sh = wb.sheet_by_name(sheet)
    if original_table_name == "sampled_sick_num5.csv":
        annotator_0 = sh.col_values(1)[1:]
        annotator_1 = sh.col_values(2)[1:]
        annotator_2 = sh.col_values(3)[1:]
        annotator_3 = sh.col_values(4)[1:]
        annotator_4 = sh.col_values(5)[1:]
        GT_value = GT_value[:500]
        annotator_0 = [int(each) for each in annotator_0[:500]]
        annotator_1 = [int(each) for each in annotator_1[:500]]
        annotator_2 = [int(each) for each in annotator_2[:500]]
        annotator_3 = [int(each) for each in annotator_3[:500]]
        annotator_4 = [int(each) for each in annotator_4[:500]]

        std = 0
        from math import sqrt
        for idx,val in enumerate(annotator_0):
            temp_mean = (annotator_0[idx] + annotator_1[idx]+ annotator_2[idx] + annotator_3[idx] + annotator_4[idx])/5
            temp_std = (pow(annotator_0[idx]-temp_mean,2) + pow(annotator_1[idx]-temp_mean,2)+ pow(annotator_2[idx]-temp_mean,2) + pow(annotator_3[idx]-temp_mean,2) + pow(annotator_4[idx]-temp_mean,2))/5
            temp_std = sqrt(temp_std)
            std += temp_std
            print(temp_std)
            # print(temp_std)

        std = std/len(GT_value)
        # print(len(GT_value))
        print("average std:{0}".format(std))

        if (len(GT_value) != len(annotator_0) or len(GT_value) != len(annotator_1) or len(GT_value) != len(annotator_2)
        or len(GT_value) != len(annotator_3) or len(GT_value) != len(annotator_4)):
            print("lengh of GT_value:{0}".format(len(GT_value)))
            print("lengh of annotator_0:{0}".format(len(annotator_0)))
            print("lengh of annotator_1:{0}".format(len(annotator_1)))
            print("lengh of annotator_1:{0}".format(len(annotator_2)))
            print("lengh of annotator_1:{0}".format(len(annotator_3)))
            print("lengh of annotator_1:{0}".format(len(annotator_4)))

            raise ValueError("length does not equal")
        for idx, val in enumerate(annotator_0):
            if (        (annotator_0[idx] == -1 or annotator_0[idx] == -2)
                    or (annotator_1[idx] == -1 or annotator_1[idx] == -2)
                    or (annotator_2[idx] == -1 or annotator_2[idx] == -2)
                    or (annotator_3[idx] == -1 or annotator_3[idx] == -2)
                    or (annotator_4[idx] == -1 or annotator_4[idx] == -2)

            ):
                annotator_0.pop(idx)
                annotator_1.pop(idx)
                annotator_2.pop(idx)
                annotator_3.pop(idx)
                annotator_4.pop(idx)
                GT_value.pop(idx)
        metric_0 = load_metric("./metrics/glue", ID_name, cache_dir="./huggingface/metrics")
        metric_0.add_batch(predictions=annotator_0, references=GT_value)
        metric_0 = metric_0.compute()

        metric_1 = load_metric("./metrics/glue", ID_name, cache_dir="./huggingface/metrics")
        metric_1.add_batch(predictions=annotator_1, references=GT_value)
        metric_1 = metric_1.compute()

        metric_2 = load_metric("./metrics/glue", ID_name, cache_dir="./huggingface/metrics")
        metric_2.add_batch(predictions=annotator_2, references=GT_value)
        metric_2 = metric_2.compute()

        metric_3 = load_metric("./metrics/glue", ID_name, cache_dir="./huggingface/metrics")
        metric_3.add_batch(predictions=annotator_3, references=GT_value)
        metric_3 = metric_3.compute()

        metric_4 = load_metric("./metrics/glue", ID_name, cache_dir="./huggingface/metrics")
        metric_4.add_batch(predictions=annotator_4, references=GT_value)
        metric_4 = metric_4.compute()

        avg_metric = {}

        for k in metric_0.keys():
            avg_metric[k] = (metric_0[k] + metric_1[k] + metric_2[k] + metric_3[k] + metric_4[k] ) / 5

        save_path = os.path.join(args["save_dir"],
                                 "{0}_results.json".format(original_table_name.split("_", 1)[1].split(".")[0]))
        final_result = {
            "annotator_0": metric_0,
            "annotator_1": metric_1,
            "annotator_2": metric_2,
            "annotator_3": metric_3,
            "annotator_4": metric_4,
            "avg_metric": avg_metric
        }
        with open(save_path, "w") as f:
            f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')
    else:
        # print(file_path)
        if ("flipkart" in file_path) or ("scitail" in file_path):
            annotator_0 = sh.col_values(-2)[1:]
            annotator_1 = sh.col_values(-1)[1:]
            def filter_func(col_values):
                new_col_values = []
                for each in col_values:
                    if each == "":
                        new_col_values.append(-1)
                    else:
                        new_col_values.append(each)
                return new_col_values
            annotator_0 = filter_func(annotator_0)
            annotator_1 =filter_func(annotator_1)
        elif "twitter" in file_path:
            annotator_0 = sh.col_values(4)[1:]
            annotator_1 = sh.col_values(5)[1:]

            def filter_func(col_values):
                new_col_values = []
                for each in col_values:
                    if each == "":
                        new_col_values.append(-1)
                    else:
                        new_col_values.append(each)
                return new_col_values

            annotator_0 = filter_func(annotator_0)
            annotator_1 = filter_func(annotator_1)
        else:
            annotator_0 = sh.col_values(1)[1:]
            annotator_1 = sh.col_values(2)[1:]

        annotator_0 = [int(each) for each in annotator_0]
        annotator_1 = [int(each) for each in annotator_1]

        # print(GT_value)
        # print(annotator_0)
        # print(annotator_1)
        if "flipkart" in file_path:
            GT_value = GT_value[:506]
        elif "scitail" in file_path:
            GT_value = GT_value[:515]
        elif "twitter" in file_path:
            GT_value = GT_value[:501]

        if ( len(GT_value) != len(annotator_0) or len(GT_value) != len(annotator_1) ):
            print("lengh of GT_value:{0}".format(len(GT_value)))
            print("lengh of annotator_0:{0}".format(len(annotator_0)))
            print("lengh of annotator_1:{0}".format(len(annotator_1)))
            raise ValueError("length does not equal")
        for idx,val in enumerate(annotator_0):
            if ( (annotator_0[idx] == -1 or annotator_1[idx] == -1) or (annotator_0[idx]==-2 or annotator_1[idx]==-2)):
                annotator_0.pop(idx)
                annotator_1.pop(idx)
                GT_value.pop(idx)
        metric_0 = load_metric("./metrics/glue", ID_name, cache_dir= "./huggingface/metrics")
        metric_0.add_batch(predictions=annotator_0, references=GT_value)
        metric_0 = metric_0.compute()

        metric_1 = load_metric("./metrics/glue", ID_name, cache_dir= "./huggingface/metrics")
        metric_1.add_batch(predictions=annotator_1, references=GT_value)
        metric_1 = metric_1.compute()

        avg_metric = {}

        for k in metric_0.keys():
            avg_metric[k] = (metric_0[k] + metric_1[k])/2

        save_path = os.path.join(args["save_dir"], "{0}_results.json".format(original_table_name.split("_",1)[1].split(".")[0]))
        final_result = {
            "annotator_0":metric_0,
            "annotator_1":metric_1,
            "avg_metric":avg_metric
        }
        with open(save_path, "w") as f:
            f.write(json.dumps(final_result, ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')

    return final_result

print(read_annotation(os.path.join(args["human_annotated_dir"], "SA_12.7.xlsx"), "imdb", "sampled_imdb.csv", "sst2"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "SA_12.7.xlsx"), "yelp", "sampled_yelp_polarity.csv", "sst2"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "SA_12.7.xlsx"), "amazon", "sampled_amazon_polarity.csv", "sst2"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "cola_ood_12.9.xlsx"), "cola", "sampled_cola_ood.csv", "cola"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "textual_entailment.xlsx"), "rte","sampled_rte.csv", "rte"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "textual_entailment.xlsx"), "qqp","sampled_qqp.csv", "rte"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "textual_entailment.xlsx"), "hans","sampled_hans.csv", "rte"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "stsb_ood.xlsx"), "stsb","sampled_sick_num5.csv", "stsb"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "qnli_ood.xlsx"), "qnli","sampled_qnli_ood.csv", "qnli"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "mnli_ood.xlsx"), "mnli-mismatched","sampled_mnli_mismatched.csv", "mnli"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "mnli_ood.xlsx"), "snli","sampled_snli.csv", "mnli"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "mnli_ood.xlsx"), "sick","sampled_sick_num3.csv", "mnli"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "sampled_flipkart.xlsx"),"Sheet1","sampled_flipkart.csv", "sst2"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "sampled_scitail.xlsx"),"Sheet1","sampled_scitail.csv", "rte"))
print(read_annotation(os.path.join(args["human_annotated_dir"], "sampled_twitter.xlsx"),"Sheet1","sampled_twitter.csv", "mrpc"))