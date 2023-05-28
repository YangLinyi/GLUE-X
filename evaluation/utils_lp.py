
from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict
import numpy as np
from datasets import disable_caching
disable_caching()
from datasets import load_metric,load_dataset,Value
from datasets import concatenate_datasets
from transformers import AutoTokenizer,DataCollatorWithPadding
from transformers import AutoModel
import torch
import time, os
import json
from tqdm import *
from utils_gluex import delete_all_pths_except_best, mode, mkdir
import pandas as pd
import datasets as da


from math import ceil
import transformers
from transformers import get_linear_schedule_with_warmup
import argparse
import random, torch

def get_checkpoint_type(checkpoint):
    checkpoints_type = {
        "roberta-base":"encoder",
        "roberta-large":"encoder",
        "xlnet-base-cased":"encoder",
        "xlnet-large-cased":"encoder",
        "albert-base-v2":"encoder",
        "bert-base-uncased":"encoder",
        "bert-large-uncased":"encoder",
        "distilbert-base-uncased":"encoder",

        "gpt2":"decoder",
        "gpt2-medium":"decoder",
        "gpt2-large":"decoder",

        "bart-base":"encoder-decoder",
        "bart-large":"encoder-decoder",

        "t5-small":"seq2seq",
        "t5-base":"seq2seq",
        "t5-large":"seq2seq",

        "electra-small-discriminator":"encoder",
        "electra-base-discriminator": "encoder",
        "electra-large-discriminator": "encoder",

    }
    return checkpoints_type[os.path.basename(checkpoint)]



def split_dict(target_dict, ratio, args, shuffle=False):
    #ratio range from 0 to 1
    keys_list = list(target_dict.keys())
    num_keys = len(keys_list)
    split_num_keys = int(num_keys*ratio)
    if shuffle:
        random.seed(args["random_seed"])
        random.shuffle(keys_list)
    split_keys_list_0 = keys_list[ :split_num_keys]
    split_keys_list_1 = keys_list[split_num_keys: ]
    split_dict_0 = {}
    split_dict_1 = {}
    for key0 in split_keys_list_0:
        split_dict_0[key0] = target_dict[key0]
    for key1 in split_keys_list_1:
        split_dict_1[key1] = target_dict[key1]

    return split_dict_0, split_dict_1





def reproduce(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return;

def get_eval_batch_size(args):
    batch_size_eval_A100 = {
        "../Models/roberta-base": 500,
        "../Models/roberta-large": 1000,
        "../Models/xlnet-large-cased": 200,
        "../Models/xlnet-base-cased": 500,
        "../Models/gpt2-medium": 1000,
        "../Models/gpt2-large": 600,
        "../Models/distilbert-base-uncased":1000,
        "../Models/albert-base-v2": 1000 ,
        "../Models/bert-base-uncased": 1000,
        "../Models/bert-large-uncased":1000,
        "../Models/electra-small-discriminator":1000,
        "../Models/electra-base-discriminator":1000,
        "../Models/electra-large-discriminator":500,
        "../Models/t5-small": 300,
        "../Models/t5-base": 200,
        "../Models/t5-large": 90,

        "../Models/gpt2": 200,
        "../Models/gpt2-medium": 200,
        "../Models/gpt2-large": 50,
        "../Models/bart-base": 200,
        "../Models/bart-large": 200,
    }
    batch_size_eval_V100={
        "../Models/roberta-base": 200,
        "../Models/roberta-large": 50,
        "../Models/xlnet-large-cased": 100,
        "../Models/xlnet-base-cased": 200,

        "../Models/distilbert-base-uncased":500,
        "../Models/albert-base-v2": 500 ,
        "../Models/bert-base-uncased": 50,
        "../Models/bert-large-uncased":30,
        "../Models/electra-small-discriminator":500,
        "../Models/electra-base-discriminator":300,
        "../Models/electra-large-discriminator":10,

        "../Models/t5-small": 100,
        "../Models/t5-base": 50,
        "../Models/t5-large": 30,

        "../Models/gpt2": 80,
        "../Models/gpt2-medium": 60,
        "../Models/gpt2-large": 30,
        "../Models/bart-base": 50,
        "../Models/bart-large": 30,
    }
    if args["device_type"] == "A100":
        batch_size_eval = batch_size_eval_A100
    elif args["device_type"] == "V100":
        batch_size_eval = batch_size_eval_V100
    else:
        raise ValueError("device type ont supported!")

    return batch_size_eval["../Models/"+args["checkpoint"]]

def parse_args():
    args = {
        "test": "",

        "finetune_from_existing_lp_results": "",
        "ftall": "",
        "linear_probing": "",

        "existing_lp_model_performance_root": "",

        "type": "",
        "task_name": "",
        "output_dir": "",
        "checkpoint": "",
        # "dataset": "",
        "num_labels": "",
        "metric_for_best_model": "",
        "greater_is_better": "",
        "lr": "",
        "lr_list": "",

        "training_epoch_lp": "",
        "training_epoch_finetune_from_existing_lp_results": "",
        "training_epoch_ftall": "",
        "training_epoch": "",

        "dropout": "",

        "per_device_train_batch_size": "",
        "per_device_eval_batch_size": "",
        "batchsize_per_device_list": "",

        "max_len": "",
        "random_seed": "",
        "device": "",
        "device_type":"",
        "model_performance_dir": "",
        "result_dir": "",

        "CHECKPOINTS_LIST":"",
        "TASK_NAMES":"",
    }

    parser = argparse.ArgumentParser("Demo of argparse")
    parser.add_argument('--device', type=int, nargs='+', default=[0,], help='list of devices ')
    parser.add_argument('--device_type', type=str, default="A100", help="device type")

    parser.add_argument('--checkpoints_list', nargs='+', help='checkpoints list ')
    parser.add_argument('--checkpoint_type', type=str, default="encoder", help="checkpoint type")

    parser.add_argument("--seq2seq",action="store_true",help="activate seq2seq mode")
    parser.add_argument('--max_target_length', type=int, default=20, help="max target length for seq2seq model")

    parser.add_argument('--task_names', nargs='+', help='task names ')
    parser.add_argument('--OOD_TASKS_SPECIFIED', type=str, help='OOD tasks specified ')

    parser.add_argument("--test",action="store_true",help="activate test mode")

    parser.add_argument("--finetune_from_existing_lp_results",action="store_true",help=" ")

    parser.add_argument("--ftall",action="store_true",help=" ")
    parser.add_argument("--do_not_save_whole_model_performance",action="store_true",help=" ")




    parser.add_argument("--linear_probing",action="store_true",help=" ")

    parser.add_argument("--continue_lp_from_existing_lp_checkpoints",action="store_true",help=" ")


    parser.add_argument("--test_OOD",action="store_true",help=" ")
    parser.add_argument("--do_not_save_whole_ood_model_performance",action="store_true",help=" ")
    parser.add_argument('--OOD_model_performance_output_dir', type=str, help="OOD_model_performance_output_dir")




    parser.add_argument('--existing_lp_model_performance_root', type=str,
                        default="./model_performance_lp_full", help="dir to existing linear probing model performance root")
    parser.add_argument('--exisiting_checkpoints_lp_performance_suffix', type=str, help="suffix of existing linear probing model performance root")


    parser.add_argument('--training_epoch_lp', type=int, default=120, help="number of epochs")
    parser.add_argument('--training_epoch_finetune_from_existing_lp_results', type=int, default=5, help="number of epochs")
    parser.add_argument('--training_epoch_ftall', type=int, default=10, help="number of epochs")
    parser.add_argument('--training_epoch_lp_continue', type=int, default=120, help="number of epochs")
    parser.add_argument('--initial_epoch', type=int, default=0, help="initial number of epochs")



    parser.add_argument('--lr_list', type=float, nargs='+', default=[2e-5, 3e-5], help='list of learning rate ')
    parser.add_argument('--batch_size_per_device_list', type=int, nargs='+', default=[8, 4], help='list of learning rate ')

    parser.add_argument('--max_len', type=int, default=512, help='max tokenize length')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--random_seed', type=int, default=2022, help='random seed')

    ###########for OOD testing############################################################

    parser.add_argument('--checkpoints_performance_dir_to_be_OOD_tested', type=str, help="root of checkpoints to be tested")
    parser.add_argument('--checkpoints_performance_suffix_to_be_OOD_tested', type=str, help="root of checkpoints to be tested")
    parser.add_argument('--OOD_performance_output_dir', type=str, help="OOD performance output root")
    parser.add_argument('--OOD_task_result_output_dir', type=str, help="OOD result root")
    parser.add_argument('--OOD_each_task_result_output_dir', type=str, help="OOD each task result root")



    ARGS = parser.parse_args()
    if ARGS.test_OOD:
        args["model_performance_dir"] = ARGS.checkpoints_performance_dir_to_be_OOD_tested
        args["model_performance_suffix"] = ARGS.checkpoints_performance_suffix_to_be_OOD_tested

    args["OOD_model_performance_output_dir"] = ARGS.OOD_model_performance_output_dir
    args["OOD_each_task_result_output_dir"] = ARGS.OOD_each_task_result_output_dir
    args["do_not_save_whole_ood_model_performance"] = ARGS.do_not_save_whole_ood_model_performance
    ###########################################################################################################



    args["device"] = ARGS.device
    args["test"] = ARGS.test
    args["finetune_from_existing_lp_results"] = ARGS.finetune_from_existing_lp_results
    args["ftall"] = ARGS.ftall
    args["linear_probing"] = ARGS.linear_probing
    args["existing_lp_model_performance_root"] = ARGS.existing_lp_model_performance_root
    args["training_epoch_lp"] = ARGS.training_epoch_lp
    args["training_epoch_finetune_from_existing_lp_results"] = ARGS.training_epoch_finetune_from_existing_lp_results
    args["training_epoch_ftall"] = ARGS.training_epoch_ftall
    args["lr_list"] = ARGS.lr_list
    args["batch_size_per_device_list"] = ARGS.batch_size_per_device_list
    args["max_len"] = ARGS.max_len
    args["dropout"] = ARGS.dropout
    args["random_seed"] = ARGS.random_seed
    args["CHECKPOINTS_LIST"] = ARGS.checkpoints_list
    args["TASK_NAMES"] = ARGS.task_names
    args["OOD_TASKS_SPECIFIED"] =ARGS.OOD_TASKS_SPECIFIED
    args["device_type"] = ARGS.device_type
    args["exisiting_checkpoints_lp_performance_suffix"] = ARGS.exisiting_checkpoints_lp_performance_suffix
    args["do_not_save_whole_model_performance"] = ARGS.do_not_save_whole_model_performance
    args["OOD_task_result_output_dir"] = ARGS.OOD_task_result_output_dir
    args["seq2seq"] = ARGS.seq2seq
    args["max_target_length"] = ARGS.max_target_length
    args["checkpoint_type"] = ARGS.checkpoint_type
    args["continue_lp_from_existing_lp_checkpoints"] = ARGS.continue_lp_from_existing_lp_checkpoints
    args["training_epoch_lp_continue"] = ARGS.training_epoch_lp_continue
    args["initial_epoch"] = ARGS.initial_epoch

    return args

if __name__ == '__main__':
    args = parse_args()
    print(args.addresses)

def preprocess_data(args):
    ###########################################load the dataset#########################################
    if args["task_name"] == "paws_labeled_final_wiki":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/paws", 'labeled_final', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "paws_labeled_swap_wiki":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/paws", 'labeled_swap', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "paws_unlabeled_final_wiki":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/paws", 'unlabeled_final', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "paws_qqp":
        data_0 = pd.read_csv("/yanglinyi/new_codes_cluster/paws/paws_qqp/output/train.tsv", sep="\t")
        data_1 = pd.read_csv("/yanglinyi/new_codes_cluster/paws/paws_qqp/output/dev_and_test.tsv", sep="\t")
        data = pd.concat([data_0, data_1], ignore_index=True)
        data = da.Dataset.from_pandas(data)
    elif args["task_name"] == "qqp":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'qqp', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "mrpc":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'mrpc', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "rte":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'rte', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "mnli_mismatched":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'mnli', split="validation_mismatched",
                            cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "sick_num5":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/sick", cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "sick_num3":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/sick", cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "hans":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/hans", cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "qnli_validation":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'qnli', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
        data = data.rename_column("question", "sentence1")
        data = data.rename_column("sentence", "sentence2")
    elif args["task_name"] == "cola_validation":
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/glue", 'cola', cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    elif args["task_name"] == "cola_ood":
        from datasets import load_from_disk
        data = load_from_disk("/yanglinyi/new_codes_cluster/cola_data/cola_ood")
    elif args["task_name"] == "qnli_ood":
        from datasets import load_from_disk
        data = load_from_disk("/yanglinyi/new_codes_cluster/qnli_data/qnli_ood")
    elif args["task_name"] == "scitail":
        # data_0 = pd.read_csv("/mnt/beegfs/inspurfs/user-fs/yanglinyi/new_codes_cluster/datasets/scitail/scitail_1.0_train.tsv", sep="\t")
        # data_1 = pd.read_csv("/mnt/beegfs/inspurfs/user-fs/yanglinyi/new_codes_cluster/datasets/scitail/scitail_1.0_dev.tsv", sep="\t")
        # data_2 = pd.read_csv("/mnt/beegfs/inspurfs/user-fs/yanglinyi/new_codes_cluster/datasets/scitail/scitail_1.0_test.tsv", sep="\t")
        # data = pd.concat([data_0, data_1, data2], ignore_index=True)
        data = pd.read_csv("/yanglinyi/new_codes_cluster/datasets/scitail/merged_scitail.tsv", sep="\t")
        data = da.Dataset.from_pandas(data)
    elif args["task_name"] == "flipkart":
        data = pd.read_csv("/yanglinyi/new_codes_cluster/datasets/flipkart/flipkart_updated.tsv", sep='\t', error_bad_lines=False)
        # data.rename(columns={'Summary': 'sentence', 'Sentiment': 'label'}, inplace=True)
        # data = data.drop(columns=['product_name',"product_price", "Rate", "Review"])
        # print(data)
        data = da.Dataset.from_pandas(data)
    elif args["task_name"] == "twitter":
        data_train = pd.read_csv("/yanglinyi/new_codes_cluster/datasets/SemEval-PIT2015-github/train.tsv", sep='\t', error_bad_lines=False)
        data_dev = pd.read_csv("/yanglinyi/new_codes_cluster/datasets/SemEval-PIT2015-github/dev.tsv", sep='\t', error_bad_lines=False)
        data_test = pd.read_csv("/yanglinyi/new_codes_cluster/datasets/SemEval-PIT2015-github/test.tsv", sep='\t', error_bad_lines=False)
        data = pd.concat([data_train, data_dev, data_test], ignore_index=True)
        data = da.Dataset.from_pandas(data)


    else:
        data = load_dataset("/yanglinyi/new_codes_cluster/datasets/" + args["task_name"], cache_dir="/yanglinyi/new_codes_cluster/huggingface/datasets")
    # print(data)

    #############preprocess the dataset#############################################################
    if args["task_name"] == "sick_num5":
        data = data.map(lambda examples: {"relatedness_score": round(examples["relatedness_score"] - 1.0)})
        data = data.rename_column("sentence_A", "sentence1")
        data = data.rename_column("sentence_B", "sentence2")
        data = data.remove_columns(
            ['id', 'label', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original',
             'sentence_A_dataset', 'sentence_B_dataset'])
        data = data.rename_column("relatedness_score", "label")
    if args["task_name"] == "sick_num3":
        data = data.rename_column("sentence_A", "sentence1")
        data = data.rename_column("sentence_B", "sentence2")
        data = data.remove_columns(
            ['id', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original',
             'sentence_A_dataset', 'sentence_B_dataset'])

    if args["task_name"] == "imdb":
        data = data.rename_column("text", "sentence")
    elif args["task_name"] == "yelp_polarity":
        data = data.rename_column("text", "sentence")
    elif args["task_name"] == "amazon_polarity":
        data = data.rename_column("content", "sentence")
        data = data.remove_columns("title")
    elif args["task_name"] == "qqp":
        if args["ID_name"] == "rte":
            data = data.map(lambda examples: {"label": -examples["label"] + 1.0})
        data = data.rename_column("question1", "sentence1")
        data = data.rename_column("question2", "sentence2")
    elif args["task_name"] == "rte":
        if args["ID_name"] != "rte":
            data = data.map(lambda examples: {"label": -examples["label"] + 1.0})
    elif args["task_name"] == "mnli_mismatched":
        data = data.rename_column("hypothesis", "sentence2")
        data = data.rename_column("premise", "sentence1")
    elif args["task_name"] == "snli":
        data = data.rename_column("hypothesis", "sentence2")
        data = data.rename_column("premise", "sentence1")

    elif args["task_name"] == "mrpc":
        if args["ID_name"] == "rte":
            data = data.map(lambda examples: {"label": -examples["label"] + 1.0})
    elif args["task_name"] == "hans":
        if args["ID_name"] != "rte":
            data = data.map(lambda examples: {"label": -examples["label"] + 1.0})
        data = data.rename_column("premise", "sentence1")
        data = data.rename_column("hypothesis", "sentence2")
        data = data.remove_columns(['parse_premise', 'parse_hypothesis', 'binary_parse_premise',
                                    'binary_parse_hypothesis', 'heuristic', 'subcase', 'template'])
    elif "paws" in args["task_name"]:
        if args["ID_name"] == "rte":
            data = data.map(lambda examples: {"label": -examples["label"] + 1.0})
    elif args["task_name"] == "qnli_ood":
        data = data.rename_column("question", "sentence1")
        data = data.rename_column("sentence", "sentence2")

        def convert_qnli_dataset(example):
            if example["label"] == "entailment":
                return 0
            elif example["label"] == "not_entailment":
                return 1
            else:
                raise ValueError("label not legal")

        data = data.map(lambda examples: {"label": convert_qnli_dataset(examples)})
    elif args["task_name"] == "scitail":
        def convert_scitail_dataset(example):
            if example["label"] == "entails":
                return 0
            elif example["label"] == "neutral":
                return 1
            else:
                raise ValueError("label not legal")
        data = data.map(lambda examples: {"label": convert_scitail_dataset(examples)})
        data = data.rename_column("premise", "sentence1")
        data = data.rename_column("hypothesis", "sentence2")
    elif args["task_name"] == "flipkart":
        def convert_flipkart_dataset(example):
            if example["label"] == "negative":
                return 0
            elif example["label"] == "neutral":
                return 0
            elif example["label"] == "positive":
                return 1
            else:
                raise ValueError("label not legal")
        data = data.rename_column("Summary", "sentence")
        data = data.rename_column("Sentiment", "label")
        data = data.map(lambda examples: {"label": convert_flipkart_dataset(examples)})
    elif args["task_name"] == "twitter":
        data = data.rename_column("Sent_1", "sentence1")
        data = data.rename_column("Sent_2", "sentence2")
        data = data.rename_column("Label", "label")

#hi shuibai
    # print(data)
    # print(data["train"][:3])
    #########################concatenate the datasets############################################
    if args["task_name"] == "imdb" or args["task_name"] == "yelp_polarity" or args["task_name"] == "amazon_polarity":
        data = concatenate_datasets([data["train"], data["test"]])
    elif args["task_name"] == "paws_labeled_final_wiki":
        data = concatenate_datasets([data["train"], data["test"], data["validation"]])
    elif args["task_name"] == "paws_labeled_swap_wiki":
        data = data["train"]
    elif args["task_name"] == "paws_unlabeled_final_wiki":
        data = concatenate_datasets([data["train"], data["validation"]])
    elif args["task_name"] == "paws_qqp":
        data = data
    elif args["task_name"] == "sick_num5":
        data = concatenate_datasets([data["train"], data["test"], data["validation"]])
    elif args["task_name"] == "sick_num3":
        data = concatenate_datasets([data["train"], data["test"], data["validation"]])
    elif args["task_name"] == "qqp":
        data = concatenate_datasets([data["train"], data["validation"]])
    elif args["task_name"] == "rte":
        data = concatenate_datasets([data["train"], data["validation"]])
    elif args["task_name"] == "mrpc":
        data = concatenate_datasets([data["train"], data["validation"]])

    elif args["task_name"] == "mnli_mismatched":
        data = data
    elif args["task_name"] == "snli":
        data = concatenate_datasets([data["train"], data["test"], data["validation"]])
    elif args["task_name"] == "hans":
        data = concatenate_datasets([data["train"], data["validation"]])
    elif args["task_name"] == "qnli_validation":
        data = data["validation"]
    elif args["task_name"] == "cola_validation":
        data = data["validation"]
    elif args["task_name"] == "cola_ood":
        data = data
    elif args["task_name"] == "qnli_ood":
        data = data

    return data




def args_preprocess(args):
    if ( int(args["finetune_from_existing_lp_results"]) + int(args["linear_probing"]) + int(args["ftall"]) + int(args["continue_lp_from_existing_lp_checkpoints"]) ) > 1:
        raise ValueError("Only one of the fintuning mode can be chose")
    if (args["seq2seq"] and args["linear_probing"]) or (args["seq2seq"] and args["finetune_from_existing_lp_results"]) or (args["seq2seq"] and args["continue_lp_from_existing_lp_checkpoints"]):
        raise ValueError("seq2seq model can not be linear probed ")
    for checkpoint in args["CHECKPOINTS_LIST"]:
        if "t5" in checkpoint:
            if not args["seq2seq"]:
                raise ValueError("You need to activate seq2seq mode")
        if ("gpt2" in checkpoint) or ("bart" in checkpoint):
            if not (args["checkpoint_type"] == "decoder"):
                raise ValueError("gpt2/bart is decoder(encoder-decoder) type, you need to specify checkpoint_type as decoder")

    if args["finetune_from_existing_lp_results"]:
        args["result_dir"] = "/yanglinyi/new_codes_cluster/results_lpthenft"
        args["model_performance_dir"] = "/yanglinyi/new_codes_cluster/model_performance_lpthenft"
        args["training_epoch"] = args["training_epoch_finetune_from_existing_lp_results"]
    elif args["linear_probing"]:
        args["result_dir"] = "/yanglinyi/new_codes_cluster/results_lp"
        args["model_performance_dir"] = "/yanglinyi/new_codes_cluster/model_performance_lp"
        args["training_epoch"] = args["training_epoch_lp"]
    elif args["continue_lp_from_existing_lp_checkpoints"]:
        args["result_dir"] = "/yanglinyi/new_codes_cluster/results_lp_continue"
        args["model_performance_dir"] = "/yanglinyi/new_codes_cluster/model_performance_lp_continue"
    elif args["ftall"]:
        args["result_dir"] = "/yanglinyi/new_codes_cluster/results_ftall"
        args["model_performance_dir"] = "/yanglinyi/new_codes_cluster/model_performance_ftall"
        args["training_epoch"] = args["training_epoch_ftall"]

    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")

    if args["test"]:
        args["result_dir"] = args["result_dir"] + "_test"
        args["model_performance_dir"] = args["model_performance_dir"] + "_test"
    else:
        args["result_dir"] = args["result_dir"] + "_full"
        args["model_performance_dir"] = args["model_performance_dir"] + "_full"

    mkdir(args["model_performance_dir"])
    mkdir(args["result_dir"])
    return args

def print_logs(args, epoch, pbar):
    if args["finetune_from_existing_lp_results"]:
        print("finetuning {0} from existing lp checkpoint".format(args["checkpoint"]))
    elif args["linear_probing"]:
        print("linear probing {0}".format(args["checkpoint"]))
    elif args["ftall"]:
        print("finetuing all the parameters of {0}".format(args["checkpoint"]))
    elif args["continue_lp_from_existing_lp_checkpoints"]:
        print("continue linear probing {0} from existing lp_checkpoints".format(args["checkpoint"]))
    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")
    pbar.set_description(
        "Training the {0}th epoch for {1} on {2} in : {3}".format(epoch + args["initial_epoch"], args["checkpoint"], args["task_name"],
                                                                  mode(args["test"])))

def get_model_path(model_save_dir, epoch, args):
    mkdir(model_save_dir)
    model_path = model_save_dir + "/" + str(epoch) + "_" + str(args["lr"]) + "_" + str(args["per_device_train_batch_size"]*len(args["device"])) #+ ".pth"
    if args["finetune_from_existing_lp_results"]:
        model_path = model_path + "_lpthenft.pth"
    elif args["linear_probing"]:
        model_path = model_path + "_lp.pth"
    elif args["ftall"]:
        model_path = model_path + "_ftall.pth"
    elif args["continue_lp_from_existing_lp_checkpoints"]:
        model_path = model_path + "_continue_lp.pth"
    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")
    return model_path

def get_log_path(model_save_dir, args):
    mkdir(model_save_dir)
    if args["task_name"] == "qnli":
        output_result_path = model_save_dir+"/training_logs_reduced_qnli_{0}_{1}".format(str(args["lr"]), str(args["per_device_train_batch_size"]*len(args["device"])))
    elif args["task_name"] == "cola":
        output_result_path = model_save_dir+"/training_logs_reduced_cola_{0}_{1}".format(str(args["lr"]), str(args["per_device_train_batch_size"]*len(args["device"])))
    else:
        output_result_path = model_save_dir + "/training_logs_{0}_{1}".format(str(args["lr"]), str(args["per_device_train_batch_size"]*len(args["device"])))

    if args["finetune_from_existing_lp_results"]:
        output_result_path = output_result_path + "_lpthenft"
    elif args["linear_probing"]:
        output_result_path = output_result_path + "_lp"
    elif args["ftall"]:
        output_result_path = output_result_path + "_ftall"
    elif args["continue_lp_from_existing_lp_checkpoints"]:
        output_result_path = output_result_path + "_continue_lp"
    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")

    if args["test"]:
        output_result_path = output_result_path +"_test.json"
    else:
        output_result_path = output_result_path + "_full.json"

    return output_result_path

def get_ood_log_path(args):
    if args["test"]:
        save_dir = os.path.join(args["OOD_task_result_output_dir"] + "_test", args["ID_name"], os.path.basename(args["checkpoint"]))
    else:
        save_dir = os.path.join(args["OOD_task_result_output_dir"] + "_full", args["ID_name"], os.path.basename(args["checkpoint"]))
    mkdir(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(args["checkpoint"]) + "_ood_" + args["ID_name"] + ".json")
    return save_path

def get_each_ood_task_log_path(args):
    if args["test"]:
        save_dir = os.path.join(args["OOD_each_task_result_output_dir"] + "_test", os.path.basename(args["checkpoint"]), args["ID_name"])
    else:
        save_dir = os.path.join(args["OOD_each_task_result_output_dir"] + "_full", os.path.basename(args["checkpoint"]), args["ID_name"])
    mkdir(save_dir)
    save_path = os.path.join(save_dir,  "ood_" + args["task_name"] + ".json")
    return save_path


def get_ood_model_performance_path(args):
    if args["test"]:
        mkdir(os.path.join(args["OOD_model_performance_output_dir"] + "_test", os.path.basename(args["checkpoint"])))
        output_path = os.path.join(args["OOD_model_performance_output_dir"] + "_test",
                                  os.path.basename(args["checkpoint"]),
                                  os.path.basename(args["checkpoint"]) + "_ood_test.json")
    else:
        mkdir(os.path.join(args["OOD_model_performance_output_dir"] + "_full", os.path.basename(args["checkpoint"])))
        output_path = os.path.join(args["OOD_model_performance_output_dir"] + "_full",
                                  os.path.basename(args["checkpoint"]),
                                  os.path.basename(args["checkpoint"]) + "_ood_full.json")
    return output_path

def get_model_performance_path(model_performance_dir, checkpoint, args):
    mkdir(model_performance_dir)
    model_performance_path = model_performance_dir + "/"  + checkpoint
    if args["finetune_from_existing_lp_results"]:
        model_performance_path = model_performance_path + "_lpthenft"
    elif args["linear_probing"]:
        model_performance_path = model_performance_path + "_lp"
    elif args["ftall"]:
        model_performance_path = model_performance_path + "_ftall"
    elif args["continue_lp_from_existing_lp_checkpoints"]:
        model_performance_path = model_performance_path + "_continue_lp"
    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")

    if args["test"]:
        model_performance_path = model_performance_path + "_test"
    else:
        model_performance_path = model_performance_path + "_full"

    model_performance_path = model_performance_path + ".json"

    return model_performance_path

def get_optimizer(model, args):
    if args["finetune_from_existing_lp_results"]:
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args["lr"],
                                     # weight_decay=args["weight_decay"]
                                     )
    elif args["linear_probing"] or args["continue_lp_from_existing_lp_checkpoints"]:
        # print(model.module.l1)
        for k, v in model.module.named_parameters():
            # print(k)
            if k.startswith("l1"):
                v.requires_grad = False  # fix the weights of the pretrained model
                # print(k)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args["lr"],
                                     # weight_decay=args["weight_decay"]
                                     )
    elif args["ftall"]:
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args["lr"],
                                     # weight_decay=args["weight_decay"]
                                     )
    else:
        raise ValueError("Fintuning method not specified, you need to select one of the finetuing method to be True")

    return optimizer


def get_best_checkpoint(log_path, args):
    with open(log_path) as f:
        training_log = json.load(f)
        epoch_performance = training_log["validation_for_each_epoch"]
        lr = str(training_log["args"]["lr"])
        batch_size = str( training_log["args"]["per_device_train_batch_size"]*len(training_log["args"]["device"]) )
    epoch_performance_avg = {}
    for k,v in epoch_performance.items():
        epoch_performance_avg[k] = get_avg_score(v)
    best_epoch = -1
    best_score = -1
    best_model_path_list = []
    for k,v in epoch_performance_avg.items():
        if v > best_score:
            # best_epoch = k
            best_score = v
    # best_model_path = best_epoch + "_"+ lr + "_" + batch_size + args["ftall_checkpoint_suffix"]
    for k,v in epoch_performance_avg.items():
        if v == best_score:
            best_epoch = k
            # best_score = v
            best_model_path = best_epoch + "_"+ lr + "_" + batch_size + args["ftall_checkpoint_suffix"]
            best_model_path_list.append(best_model_path)
    return {"best_avg_score":epoch_performance_avg[best_epoch], "best_metric": epoch_performance[best_epoch], "best_model_path_list":best_model_path_list}

def get_best_checkpoint_from_all_logs(candidates_list):
    best_avg_score = -1
    best_model_path_list = []
    best_metric = ""
    for candiadate in candidates_list:
        if candiadate["best_avg_score"] > best_avg_score:
            best_avg_score = candiadate["best_avg_score"]
            best_metric = candiadate["best_metric"]

    for candiadate in candidates_list:
        if candiadate["best_avg_score"] == best_avg_score:
            best_model_path_list += candiadate["best_model_path_list"]

    return best_avg_score, best_metric, best_model_path_list


def get_avg_score(result_dict):
    sum = 0
    num = 0
    # print(result_dict)
    for k,v in result_dict.items():
        sum += v
        num += 1
    task_score = sum / num
    if np.isnan(task_score):
        return 0

    return task_score
def get_ID_task_avg_score(result_dict):
    sum = 0
    num = 0
    for k,v in result_dict.items():
        if k=="best_model_dir":
            continue
        sum += v
        num += 1

    return sum/num



