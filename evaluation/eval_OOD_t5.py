
from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict, calcuate_accu_for_str_list, map_seq_to_value, mode
import numpy as np
import datasets as da
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

import transformers
from torch.utils.data import Dataset, DataLoader
from utils_lp import preprocess_data
def preprocess_data_t5(args):
    data = preprocess_data(args)
####################################################add prefix to the specified ID+_task##################################################
    if args["ID_name"] == "sst2":
        def add_prefix_sst2(example):
            example["sentence"] = "sst2 sentence: " + example["sentence"]
            return example
        data = data.map(add_prefix_sst2)
    elif args["ID_name"] == "cola":
        def add_prefix_cola(example):
            example["sentence"] = "cola sentence: " + example["sentence"]
            return example
        data = data.map(add_prefix_cola)
    elif args["ID_name"] == "mrpc":
        def add_prefix_mrpc(example):
            example["sentence1"] = "mrpc sentence1: " + example["sentence1"]
            example["sentence2"] = "sentence2: " + example["sentence2"]
            return example
        data = data.map(add_prefix_mrpc)
    elif args["ID_name"] == "stsb":
        def add_prefix_stsb(example):
            example["sentence1"] = "stsb sentence1: " + example["sentence1"]
            example["sentence2"] = "sentence2: " + example["sentence2"]
            return example
        data = data.map(add_prefix_stsb)
    elif args["ID_name"] == "qqp":
        def add_prefix_qqp(example):
            example["sentence1"] = "qqp question1: " + example["sentence1"]
            example["sentence2"] = "question2: " + example["sentence2"]
            return example
        data = data.map(add_prefix_qqp)
    elif args["ID_name"] == "mnli" or args["ID_name"] == "mnli_matched" or args["ID_name"] == "mnli_mismatched":
        def add_prefix_mnli(example):

            temp_sentence1 = "mnli hypothesis: " + example["sentence2"]
            # example["sentence1"] = "mnli hypothesis: " + example["sentence2"]
            example["sentence2"] = "premise: " + example["sentence1"]
            example["sentence1"] = temp_sentence1

            return example
        data = data.map(add_prefix_mnli)
    elif args["ID_name"] == "qnli":
        def add_prefix_qnli(example):
            example["sentence1"] = "qnli question: " + example["sentence1"]
            example["sentence2"] = "sentence: " + example["sentence2"]
            return example
        data = data.map(add_prefix_qnli)
    elif args["ID_name"] == "rte":
        def add_prefix_rte(example):
            example["sentence1"] = "rte sentence1: " + example["sentence1"]
            example["sentence2"] = "sentence2: " + example["sentence2"]
            return example
        data = data.map(add_prefix_rte)
    elif args["ID_name"] == "wnli":
        def add_prefix_rte(example):
            example["sentence1"] = "wnli sentence1: " + example["sentence1"]
            example["sentence2"] = "sentence2: " + example["sentence2"]
            return example
        data = data.map(add_prefix_rte)
    else:
        raise ValueError("task_name not specified in def:preprocess_data")

    return data

class Classifier(torch.nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.l1 = T5ForConditionalGeneration.from_pretrained("../Models/"+os.path.basename(args["checkpoint"]))

    def forward(self, input_ids, attention_mask, labels):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)#decoder_input_ids is automatically created by prepending a decoder_start_token_id to labels

        return output

class ood_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, args):
        self.max_len = args["max_len"]
        self.ID_name = args["ID_name"]
        self.data = dataframe

        if args["test"]:
            self.data = self.data.shuffle(seed=args["random_seed"]).select(range(min([len(self.data), 6000])))
        if args["type"] == "pair":
            self.data = self.data.map(lambda examples: tokenizer(examples['sentence1'], examples['sentence2'],
                                                                 truncation=True, add_special_tokens=True,
                                                                 max_length=self.max_len, pad_to_max_length=True,
                                                                 return_token_type_ids=True, ),
                                      batched=True,
                                      load_from_cache_file=False)
            self.data = self.data.remove_columns(["sentence1", "sentence2"])
        elif args["type"] == "single":
            self.data = self.data.map(lambda examples: tokenizer(examples["sentence"],
                                                                 truncation=True, add_special_tokens=True,
                                                                 max_length=self.max_len, pad_to_max_length=True,
                                                                 return_token_type_ids=True, ),
                                      batched=True,
                                      load_from_cache_file=False)
            self.data = self.data.remove_columns("sentence")

    def __getitem__(self, index):
        inputs = self.data[index]
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        targets = inputs['label']
        if self.ID_name == "sst2":
            if targets == 0:
                targets = "negative"
            elif targets == 1:
                targets = "positive"

        elif self.ID_name =="cola":
            if targets == 0:
                targets = "unacceptable"
            elif targets == 1:
                targets = "acceptable"

        elif self.ID_name =="mrpc":
            if targets == 0:
                targets = "not_equivalent"
            elif targets == 1:
                targets = "equivalent"

        elif self.ID_name =="stsb":
            targets = str(targets)

        elif self.ID_name =="qqp":
            if targets == 0:
                targets = "not_duplicate"
            elif targets == 1:
                targets = "duplicate"

        elif self.ID_name =="mnli":
            if targets == 0:
                targets = "entailment"
            elif targets == 1:
                targets = "neutral"
            elif targets ==2:
                targets = "contradiction"

        elif self.ID_name =="qnli":
            if targets == 0:
                targets = "entailment"
            elif targets == 1:
                targets = "not_entailment"

        elif self.ID_name =="rte":
            if targets == 0:
                targets = "entailment"
            elif targets == 1:
                targets = "not_entailment"
        elif self.ID_name =="wnli":
            if targets == 0:
                targets = "not_entailment"
            elif targets == 1:
                targets = "entailment"
        else:
            raise ValueError("task_name not specified in def:get_item in glue_dataset")
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': targets
        }

    def __len__(self):
        return len(self.data)


def Evaluation(args, model, testing_loader, tokenizer):
    model.eval()
    n_correct = 0;
    nb_tr_steps = 0;
    nb_tr_examples = 0
    metric = load_metric("./metrics/glue", args["ID_name"], cache_dir="./huggingface/metrics")

    i = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            torch.cuda.empty_cache()
            ids = data['ids'].to(args["device"][0], dtype=torch.long)
            mask = data['mask'].to(args["device"][0], dtype=torch.long)
            targets = data['targets']
            output_sequences = model.module.l1.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,  # disable sampling to test if batching affects output
            )
            output_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            temp = calcuate_accu_for_str_list(output_sequences, targets)

            n_correct += temp

            print("Batch{0} accuracy:{1}".format(i,temp/len(targets)))


            nb_tr_steps += 1
            nb_tr_examples += len(targets)
            prediction_value = map_seq_to_value(output_sequences, args["ID_name"])
            targets_value = map_seq_to_value(targets, args["ID_name"])

            metric.add_batch(predictions=prediction_value, references=targets_value)
            i += 1


    epoch_accu = (n_correct * 100) / nb_tr_examples
    print("Model: "+os.path.basename(args["checkpoint"])+"ID_task: "+ args["ID_name"]+f"  Validation Accuracy: {epoch_accu} for "+args["task_name"]+" in "+mode(args["test"]))
    return metric.compute()

def predict_by_best_model(args):
    tokenizer = T5Tokenizer.from_pretrained(args["checkpoint"])

    data = preprocess_data_t5(args)
    testing_set = ood_dataset(data, tokenizer, args)
    testing_set_length = len(testing_set)

    eval_params = {'batch_size': args["per_device_eval_batch_size"],
                   'shuffle': True,
                   'num_workers': 0
                   }
    testing_loader = DataLoader(testing_set, **eval_params)

    best_model_dir = args["best_model_dir"]

    loaded_whole_model = torch.load(best_model_dir)
    loaded_model = Classifier(args)

    loaded_model_dict = loaded_whole_model.module.state_dict()
    loaded_model.load_state_dict(loaded_model_dict)

    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=args["device"])
    model = loaded_model.cuda(device=args["device"][0])


    best_result = Evaluation(args, model, testing_loader, tokenizer)
    del model
    return ( best_result, len(testing_set) )



