
import torch
print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True

from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict, mode
import numpy as np
import datasets as da
import pandas as pd
from datasets import disable_caching
disable_caching()
from datasets import load_metric,load_dataset,Value
from datasets import concatenate_datasets

from transformers import AutoTokenizer,DataCollatorWithPadding
from transformers import AutoModel
import torch
import time, os
os.environ['TRANSFORMERS_CACHE'] = '/yanglinyi/new_codes_cluster/huggingface'
import json
from tqdm import *
from math import ceil

import transformers
from torch.utils.data import Dataset, DataLoader

from utils_lp import preprocess_data

class ood_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, args):
        self.max_len = args["max_len"]
        self.data = dataframe
        if args["test"]:
            print(self.data)
            self.data = self.data.shuffle(seed=args["random_seed"]).select(range(min([len(self.data), 6000])))
            print(self.data)

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
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(int(targets), dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)
from eval_multi_class import Classifier_encoder
from eval_multi_class import Classifier_decoder

def Evaluation(args, model, testing_loader):
    model.eval()
    n_correct = 0; nb_tr_steps = 0; nb_tr_examples = 0
    metric = load_metric("/yanglinyi/new_codes_cluster/metrics/glue", args["ID_name"], cache_dir= "/yanglinyi/new_codes_cluster/huggingface/metrics")

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            torch.cuda.empty_cache()
            ids = data['ids'].to(args["device"][0], dtype=torch.long)
            mask = data['mask'].to(args["device"][0], dtype=torch.long)
            targets = data['targets'].to(args["device"][0], dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            if outputs.data.ndim == 1:
                outputs.data = outputs.data.unsqueeze(0)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            metric.add_batch(predictions=big_idx, references=targets)

    epoch_accu = (n_correct * 100) / nb_tr_examples
    print("Model: "+os.path.basename(args["checkpoint"])+" ID_task: "+ args["ID_name"]+f"  Validation Accuracy: {epoch_accu} for "+args["task_name"]+" in "+mode(args["test"]))
    return metric.compute()

def predict_by_best_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
    if "gpt2" in args["checkpoint"]:
        tokenizer.pad_token = tokenizer.eos_token  # for gpt2 only
    data = preprocess_data(args)
    testing_set = ood_dataset(data, tokenizer, args)
    eval_params = {'batch_size': args["per_device_eval_batch_size"],
                   'shuffle': True,
                   'num_workers': 0
                   }
    testing_loader = DataLoader(testing_set, **eval_params)

    best_model_dir = args["best_model_dir"]
    loaded_whole_model = torch.load(best_model_dir)
    if args["checkpoint_type"] == "encoder":
        loaded_model = Classifier_encoder(args)
    elif args["checkpoint_type"] == "decoder":
        loaded_model = Classifier_decoder(args)
    else:
        raise ValueError("checkpoint type not supported!")
    loaded_model_dict = loaded_whole_model.module.state_dict()
    loaded_model.load_state_dict(loaded_model_dict)

    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=args["device"])
    model = loaded_model.cuda(device=args["device"][0])

    best_result = Evaluation(args, model, testing_loader)
    del model
    return ( best_result, len(testing_set) )




def predict_by_best_model_api(model, args):
    tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
    if "gpt2" in args["checkpoint"]:
        tokenizer.pad_token = tokenizer.eos_token  # for gpt2 only
    data = preprocess_data(args)
    testing_set = ood_dataset(data, tokenizer, args)
    eval_params = {'batch_size': args["per_device_eval_batch_size"],
                   'shuffle': True,
                   'num_workers': 0
                   }
    testing_loader = DataLoader(testing_set, **eval_params)

    if args["training_mode"] == "from_finetuned":
        loaded_whole_model = model
        if args["checkpoint_type"] == "encoder":
            loaded_model = Classifier_encoder(args)
        elif args["checkpoint_type"] == "decoder":
            loaded_model = Classifier_decoder(args)
        else:
            raise ValueError("checkpoint type not supported!")
        loaded_model_dict = loaded_whole_model.module.state_dict()
        loaded_model.load_state_dict(loaded_model_dict)

        loaded_model = torch.nn.DataParallel(loaded_model, device_ids=args["device"])
        model = loaded_model.cuda(device=args["device"][0])
    elif args["training_mode"] == "from_pretrained":
        model = model
    else:
        raise ValueError("training mode not supported")
    best_result = Evaluation(args, model, testing_loader)
    del model
    return ( best_result, len(testing_set) )



