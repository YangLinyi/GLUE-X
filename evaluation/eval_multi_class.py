from utils_gluex import mkdir, calcuate_accu, filter_nan_for_dict
import numpy as np
from datasets import disable_caching
disable_caching()
import os
os.environ['PYTHON_EGG_CACHE'] = './cache_backup'
os.environ['TRANSFORMERS_CACHE'] = './huggingface'
from datasets import load_metric,load_dataset,Value
from transformers import AutoTokenizer,DataCollatorWithPadding
from transformers import AutoModel
import torch
import time, os
import json
from tqdm import *
from utils_gluex import delete_all_pths_except_last_and_best, mode
from math import ceil

import transformers
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

def preprocess_data(args):
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
    if args["task_name"] == "qnli":
        data = data.rename_column("question", "sentence1")
        data = data.rename_column("sentence", "sentence2")
    return data


class glue_dataset(Dataset):
    def __init__(self, split, dataframe, tokenizer, args):
        self.data = dataframe
        self.max_len = args["max_len"]
        if args["test"]:
            if split == "train":
                self.data = self.data.shuffle(seed=args["random_seed"]).select(range(min([len(self.data), 800])))
            elif split == "validation":
                self.data = self.data.shuffle(seed=args["random_seed"])#.select(range(min([len(self.data), 200])))
        if args["type"] == "pair":
            print(self.data)
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
        self.l1 = AutoModel.from_pretrained("../Models/"+os.path.basename(args["checkpoint"]))
        if "gpt2" in args["checkpoint"]:
            self.l1.config.pad_token_id = self.l1.config.eos_token_id  # for gpt2 only
        self.pre_classifier = torch.nn.Linear(self.l1.config.hidden_size, self.l1.config.hidden_size)
        self.dropout = torch.nn.Dropout(args["dropout"])
        self.classifier = torch.nn.Linear(self.l1.config.hidden_size, args["num_labels"])

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]  # choose the first embedding

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output


class Classifier_encoder(torch.nn.Module):
    def __init__(self,args):
        super(Classifier_encoder,self).__init__()
        self.l1 = AutoModel.from_pretrained("/yanglinyi/Models/"+os.path.basename(args["checkpoint"]))
        if "gpt2" in args["checkpoint"]:
            self.l1.config.pad_token_id = self.l1.config.eos_token_id  # for gpt2 only
        self.pre_classifier = torch.nn.Linear(self.l1.config.hidden_size, self.l1.config.hidden_size)
        self.dropout = torch.nn.Dropout(args["dropout"])
        self.classifier = torch.nn.Linear(self.l1.config.hidden_size, args["num_labels"])

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]

        pooler = hidden_state[:, 0]  # choose the first embedding

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output


class Classifier_decoder(torch.nn.Module):
    def __init__(self, args):
        super(Classifier_decoder, self).__init__()
        self.l1 = AutoModel.from_pretrained("/yanglinyi/Models/" + os.path.basename(args["checkpoint"]))
        self.max_len = args["max_len"]
        self.device = args["device"]
        if "gpt2" in args["checkpoint"]:
            self.l1.config.pad_token_id = self.l1.config.eos_token_id  # for gpt2 only

        self.pre_classifier = torch.nn.Linear(self.l1.config.hidden_size, self.l1.config.hidden_size)
        self.dropout = torch.nn.Dropout(args["dropout"])
        self.classifier = torch.nn.Linear(self.l1.config.hidden_size, args["num_labels"])

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        not_ignore = input_ids.ne(self.l1.config.pad_token_id)
        last_token_pos = torch.sum(not_ignore, dim=-1)
        last_token_pos -= 1
        last_token_pos = last_token_pos.unsqueeze(1)

        hidden_state = output_1[0]  # last_hidden_state
        batch_size = hidden_state.shape[0]
        temp = torch.tensor(range(self.max_len)).unsqueeze(0).repeat(batch_size, 1).to(last_token_pos.device,
                                                                                       dtype=torch.long)  # .to(self.device[0], dtype=torch.long)
        mask = temp == last_token_pos
        pooler = hidden_state[mask, :]

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output


def Evaluation(args, model, testing_loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    n_correct = 0; tr_loss = 0; nb_tr_steps = 0; nb_tr_examples = 0
    metric = load_metric("./metrics/glue", args["task_name"], cache_dir= "./huggingface/metrics")


    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            torch.cuda.empty_cache()
            ids = data['ids'].to(args["device"][0], dtype=torch.long)
            mask = data['mask'].to(args["device"][0], dtype=torch.long)
            targets = data['targets'].to(args["device"][0], dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            metric.add_batch(predictions=big_idx, references=targets)

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    return metric.compute()



def train(args, model, epoch, training_loader,
          # optimizer,
          # scheduler
          ):
    model.train()
    tr_loss = 0; n_correct = 0; nb_tr_steps = 0; nb_tr_examples = 0
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args["lr"],
                                 # weight_decay=args["weight_decay"]
                                 )


    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(args["device"][0], dtype=torch.long)
        mask = data['mask'].to(args["device"][0], dtype=torch.long)
        targets = data['targets'].to(args["device"][0], dtype=torch.long)

        outputs = model(ids, mask)
        
        loss = loss_function(outputs, targets)
        # print(loss)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()
        # scheduler.step()

    print(f'The Total Accuracy on training set for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss on training set Epoch: {epoch_loss}")
    print(f"Training Accuracy on training set Epoch: {epoch_accu}")
    return model#, optimizer, scheduler
