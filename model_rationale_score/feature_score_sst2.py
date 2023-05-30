import os.path
import nltk
nltk.download('punkt')
import sys
sys.path.append("../evaluation")
import sys
from tqdm import tqdm_notebook
import numpy as np

import random

from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch

from sklearn.utils import shuffle
import transformers
import json



from datasets import load_metric,load_dataset,Value
import csv


import nltk
# nltk.data.path.append('D:\\python_pkg_data\\nltk_data')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import ast
import glob
import shutil

import importlib
from torch.utils.data import DataLoader
from torch.nn import Softmax
from termcolor import colored
from itertools import groupby
from operator import itemgetter
import html
from IPython.core.display import display, HTML
import more_itertools as mit
from tqdm import tqdm_notebook


def html_escape(text):
    return html.escape(text)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#t5 only
class CustomerDataset_t5(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        if self.labels[idx] == 0:
            item["labels"] = "negative"
        elif self.labels[idx] == 1:
            item["labels"] = "positive"
        # print(item)
        return item

    def __len__(self):
        return len(self.labels)

def map_label_to_idx(label, args):
        return args["tokenizer"](label).input_ids[0]

def get_rationale_spans_t5(model, text, label, args, topk=15,):
    token_text = word_tokenize(text)

    candidates, remove_terms = identify_important_terms(token_text, text)

    candidates = [ "sst2 sentence: " + candidate for candidate in candidates]

    candidates_label = [label] * len(candidates)

    candidates_encodings = args['tokenizer'](candidates, truncation = True, max_length=args["max_len"], pad_to_max_length=True,)

    candidates_dataset = CustomerDataset_t5(candidates_encodings, candidates_label)
    candidates_dataloader = DataLoader(candidates_dataset, batch_size=50, shuffle=False)

    model.eval()
    output_logits = []

    for batch in tqdm_notebook(candidates_dataloader):
        batch["input_ids"] = batch["input_ids"].cuda(args["gpu_device"])
        batch["attention_mask"] = batch["attention_mask"].cuda(args["gpu_device"])
        with torch.no_grad():
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            targets = batch["labels"]
            targets_encoding = args["tokenizer"](targets,
                                         pad_to_max_length=True,
                                         max_length=args["max_target_length"],
                                         truncation=True,
                                         add_special_tokens=True,
                                         return_tensors="pt"
                                         ).input_ids
            targets_encoding[targets_encoding == args["tokenizer"].pad_token_id] = -100
            targets_encoding = targets_encoding.to(args["gpu_device"], dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets_encoding)

            logits = outputs.logits
            logits_first_token = logits[:,0,:]

            label_index = map_label_to_idx(targets[0], args)
            output_logits.append(logits_first_token)

    outputs = torch.cat(output_logits)
    sm = Softmax(dim=1)
    outputs = sm(outputs)

    results = {}
    for idx, score in enumerate(outputs[:-1]):
        changes = abs(float(outputs[idx][label_index] - outputs[-1][label_index]))
        results[idx] = changes

    token_id = list(dict(sorted(results.items(), key=lambda item: item[1], reverse=True)).keys())

    inferred_spans = []
    for ids in token_id[:topk]:
        #         print(ids, remove_terms[ids]['terms'])
        span = [i for i in range(remove_terms[ids]['start_token'], remove_terms[ids]['end_token'])]
        inferred_spans.append(span)

    inferred_pos = []
    for span in inferred_spans:
        for number in span:
            inferred_pos.append(number)

    inferred_pos = list(set(inferred_pos))

    return inferred_pos



def import_data(directory):
    data_pos = []
    data_neg = []
    with open(directory, errors='ignore') as file:
        file = csv.reader(file, delimiter="\t")
        for idx, row in enumerate(file):
            if idx != 0:
                if row[0] == 'Negative':
                    data_neg.append({'idx': idx, 'text': row[1], 'label': 0})
                else:
                    data_pos.append({'idx': idx, 'text': row[1], 'label': 1})

    return data_neg, data_pos


def import_paired_data(directory, original_texts):
    paired_data = {}
    with open(directory, errors='ignore') as file:
        file = csv.reader(file, delimiter="\t")
        for idx, row in enumerate(file):

            if idx != 0:
                if row[2] not in paired_data.keys():
                    paired_data[row[2]] = []

                    if row[1] in original_texts:
                        paired_data[row[2]].append(
                            {'text': row[1], 'label': 0 if row[0] == 'Negative' else 1, 'ori_flag': 1})

                    else:
                        paired_data[row[2]].append(
                            {'text': row[1], 'label': 0 if row[0] == 'Negative' else 1, 'ori_flag': 0})
                else:
                    if row[1] in original_texts:
                        paired_data[row[2]].append(
                            {'text': row[1], 'label': 0 if row[0] == 'Negative' else 1, 'ori_flag': 1})
                    else:
                        paired_data[row[2]].append(
                            {'text': row[1], 'label': 0 if row[0] == 'Negative' else 1, 'ori_flag': 0})

    return paired_data


def visualise_rationales(original, rationale_spans, rationale_pos, visualise_all=False):
    if visualise_all:
        highlighted = []
        for idx, term in enumerate(word_tokenize(original)):
            if idx in rationale_pos:
                highlighted.append(colored(term, 'blue'))
            else:
                highlighted.append(term)

        return TreebankWordDetokenizer().detokenize(highlighted)

    else:
        highlights = []
        for span in rationale_spans:
            highlighted = []
            for idx, term in enumerate(word_tokenize(original)):
                if idx in span:
                    highlighted.append(colored(term, 'blue'))
                else:
                    highlighted.append(term)

            highlights.append(TreebankWordDetokenizer().detokenize(highlighted))

        return highlights


def visualise_rationales_html(original, rationale_spans, rationale_pos, visualise_all=False):
    if visualise_all:
        highlighted = []
        for idx, term in enumerate(word_tokenize(original)):
            if idx in rationale_pos:
                highlighted.append('<font color="blue">' + html_escape(term) + '</font>')
            else:
                highlighted.append(term)

        return TreebankWordDetokenizer().detokenize(highlighted)

    else:
        highlights = []
        for span in rationale_spans:
            highlighted = []
            for idx, term in enumerate(word_tokenize(original)):
                if idx in span:
                    highlighted.append('<font color="blue">' + html_escape(term) + '</font>')
                else:
                    highlighted.append(term)

            highlights.append(TreebankWordDetokenizer().detokenize(highlighted))

        return highlights


def detect_rationale_spans(non_rationale_pos, text_length, max_length=1):
    rationale_spans = []

    rationale_pos = list(set([i for i in range(text_length)]) - set(non_rationale_pos))
    rationale_pos.sort()

    for k, g in groupby(enumerate(rationale_pos), lambda ix: ix[0] - ix[1]):
        span = list(map(itemgetter(1), g))
        if len(span) <= max_length:
            rationale_spans.append(span)

    return rationale_spans, rationale_pos


def identify_important_terms(token_text, text):
    candidates = []
    remove_terms = {}
    count = 0
    for idx, token in enumerate(token_text):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms': duplicate[idx], 'start_token': idx, 'end_token': idx + 1}
        del duplicate[idx]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))

    for idx, token in enumerate(token_text[:-1]):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms': duplicate[idx:idx + 2], 'start_token': idx, 'end_token': idx + 2}
        del duplicate[idx:idx + 2]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))

    for idx, token in enumerate(token_text[:-2]):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms': duplicate[idx:idx + 3], 'start_token': idx, 'end_token': idx + 3}
        del duplicate[idx:idx + 3]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))

    candidates.append(text)

    return candidates, remove_terms


def get_rationale_spans(model, text, label, args, topk=15):

    token_text = word_tokenize(text)

    candidates, remove_terms = identify_important_terms(token_text, text)


    candidates_label = [label] * len(candidates)

    candidates_encodings = args['tokenizer'](candidates, truncation=True, max_length=args["max_len"], pad_to_max_length=True,)

    candidates_dataset = CustomerDataset(candidates_encodings, candidates_label)
    candidates_dataloader = DataLoader(candidates_dataset,
                                       # batch_size=batch_size_eval[os.path.join("../Models",args["checkpoint"])],
                                       batch_size=100,

                                       shuffle=False)

    model.eval()
    output_logits = []
    for batch in tqdm_notebook(candidates_dataloader):
        batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
        #         print(batch)
        with torch.no_grad():
            #             outputs = model(**batch)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predictions = torch.argmax(logits, dim=-1)
        output_logits.append(logits)

    outputs = torch.cat(output_logits)
    sm = Softmax(dim=1)
    outputs = sm(outputs)

    results = {}
    for idx, score in enumerate(outputs[:-1]):
        changes = abs(float(outputs[idx][label] - outputs[-1][label]))
        results[idx] = changes

    token_id = list(dict(sorted(results.items(), key=lambda item: item[1], reverse=True)).keys())

    inferred_spans = []
    for ids in token_id[:topk]:
        span = [i for i in range(remove_terms[ids]['start_token'], remove_terms[ids]['end_token'])]
        inferred_spans.append(span)

    inferred_pos = []
    for span in inferred_spans:
        for number in span:
            inferred_pos.append(number)

    inferred_pos = list(set(inferred_pos))

    return inferred_pos

def metrics(pos, pos_GT):
    correct = 0

    for p in pos:
        if p in pos_GT:
            correct += 1

    if correct == 0:
        return 0, 0, 0
    precision = correct / len(pos)
    recall = correct / len(pos_GT)

    f1 = 2 * precision * recall / (precision + recall)


    return f1, precision, recall

def feature_score_sst2_api(model, testing_dataset, args):

    args["gpu_device"] = args["device"][0]
    average_f1 = 0
    average_precision = 0
    average_recall = 0

    for key in testing_dataset.keys():
        # print(key)
        text = testing_dataset[key]['original_text']
        label = testing_dataset[key]['label']

        if "t5" in args["checkpoint"]:
            args["tokenizer"] = T5Tokenizer.from_pretrained(os.path.join("/yanglinyi/Models", args["checkpoint"]))
        else:
            args["tokenizer"] = transformers.AutoTokenizer.from_pretrained(os.path.join("/yanglinyi/Models", args["checkpoint"]) )
        if "gpt2" in args["checkpoint"]:
            args["tokenizer"].pad_token = args["tokenizer"].eos_token  # for gpt2 only
        if "t5" in args["checkpoint"]:
            generated_rationales_spans = get_rationale_spans_t5(model, text, label, args)
        else:
            generated_rationales_spans = get_rationale_spans(model, text, label, args)


        f1, precision, recall = metrics(pos=generated_rationales_spans, pos_GT=testing_dataset[key]["rationale_position"])
        average_f1 += f1
        average_recall += recall
        average_precision += precision


    average_f1 = average_f1 / len(list(testing_dataset.keys()))
    average_precision = average_precision / len(list(testing_dataset.keys()))
    average_recall = average_recall / len(list(testing_dataset.keys()))

    print("average_f1:{0}".format(str(average_f1)))

    return average_f1, average_precision, average_recall




def feature_score(model_dir, target_dataset, args, Original_Text, Rationale_Positions):
    mkdir(os.path.join(os.path.join(args["model_rationale_save_dir"], target_dataset)))
    with open(os.path.join(args["model_rationale_save_dir"], target_dataset, args["checkpoint"] + "_rationale.csv"), 'w',
              newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text",
                          "model_rationale_pos",
                          "human_rationale_pos_1",
                         "human_rationale_pos_2",
                         "human_rationale_overlap"
        ])

    model = torch.load(model_dir).module.cuda(args['gpu_device'])
    average_f1 = {}
    average_precision = {}
    average_recall = {}
    for labeler in Rationale_Positions.keys():
        average_f1[labeler] = 0
        average_precision[labeler] = 0
        average_recall[labeler] = 0

    original_text = Original_Text[target_dataset]

    for key in list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()):
        text = original_text[key]['text']
        label = original_text[key]['label']
        token_text = word_tokenize(text)
        if "t5" in model_dir:
            args["tokenizer"] = T5Tokenizer.from_pretrained(os.path.join("../../Models", model_dir.split("/")[-2]))
        else:
            args["tokenizer"] = transformers.AutoTokenizer.from_pretrained(os.path.join("../../Models", model_dir.split("/")[-2]) )
        if "gpt2" in model_dir:
            args["tokenizer"].pad_token = args["tokenizer"].eos_token  # for gpt2 only
        if "t5" in model_dir:
            generated_rationales_spans = get_rationale_spans_t5(model, text, label, args)
        else:
            generated_rationales_spans = get_rationale_spans(model, text, label, args)

        for labeler in Rationale_Positions.keys():
            f1, precision, recall = metrics(pos=generated_rationales_spans, pos_GT=Rationale_Positions[labeler][target_dataset][key])
            average_f1[labeler] += f1
            average_recall[labeler] += recall
            average_precision[labeler] += precision

        with open(os.path.join(args["model_rationale_save_dir"], target_dataset, args["checkpoint"] + "_rationale.csv"), 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            sorted_generated_rationales_spans = sorted(generated_rationales_spans)
            sorted_human_rationale_pos_1= sorted(Rationale_Positions[args["labeler"][0]][target_dataset][key])
            sorted_human_rationale_pos_2= sorted(Rationale_Positions[args["labeler"][1]][target_dataset][key])
            sorted_human_rationale_pos_overlap= sorted(Rationale_Positions["overlap"][target_dataset][key])

            writer.writerow([original_text[key]["text"].encode("utf-8"),
                             sorted_generated_rationales_spans,
                             sorted_human_rationale_pos_1, sorted_human_rationale_pos_2, sorted_human_rationale_pos_overlap
                             ])


    for labeler in Rationale_Positions.keys():
        average_f1[labeler] = average_f1[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))
        average_precision[labeler] = average_precision[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))
        average_recall[labeler] = average_recall[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))


    score_save_dir = os.path.join(args["output_dir"], args["checkpoint"], target_dataset)
    mkdir(score_save_dir)
    score_save_path = os.path.join(score_save_dir, target_dataset+".json")
    with open(score_save_path, "w") as f:
        f.write(json.dumps({
            "dataset_name":target_dataset,
            "f1_score":average_f1,
            "precision":average_precision,
            "recall":average_recall,
            "dataset_length":len(list( Rationale_Positions[args["labeler"][0]][target_dataset].keys() )),
        }
            , ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')


from datasets import load_from_disk
def new_Text_preprocess(args):
    original_text = {}
    for target_dataset in args["new_GT_rationale_datasets"]:
        original_text[target_dataset] = {}
        target_dataset_path = os.path.join(args["new_GT_rationales_dir"], target_dataset)
        loaded_dataset = load_from_disk(target_dataset_path)
        for idx,row in enumerate(loaded_dataset):
            original_text[target_dataset][str(idx)] = {"text":row["sentence"], "label":row["label"]}

    return original_text


def get_rationale_positions_GT(rationale_path):
    rationales = json.load(open(rationale_path, 'r', encoding="utf-8"))

    rationale_spans = {}
    for item in rationales:
        key = list(item.keys())[1]
        index = key.split("_")[1]
        rationale_spans[index] = item[key]

    rationale_positions = {}

    train_keys = list(rationale_spans.keys())

    for key in train_keys:
        doc_positions = []
        positions = rationale_spans[key]
        for span in positions:
            start = span['start_token']
            end = span['end_token']
            doc_positions = doc_positions + [i for i in range(start, end)]

        rationale_positions[key] = doc_positions

    return rationale_positions

import re
def new_GT_preprocess(args):#labler can be "1" or "2"
    rationale_positions = {}
    for labeler in args["labeler"]:
        rationale_positions[labeler] = {}
        for target_dataset in args["new_GT_rationale_datasets"]:
            rationale_positions[labeler][target_dataset] = {}
            target_dataset_path_list = []
            for file in os.listdir(args["new_GT_rationales_dir"]):
                if (target_dataset in file) and ("batch" in file):
                    for each_json_file in os.listdir(os.path.join(args["new_GT_rationales_dir"], file)):
                        if each_json_file.split(".js")[0].split("_")[-1] == labeler:
                            target_dataset_path_list.append(os.path.join(args["new_GT_rationales_dir"], file, each_json_file))


            for each_batch_path in target_dataset_path_list:
                temp_rationale_positions = get_rationale_positions_GT(each_batch_path)
                for k,v in temp_rationale_positions.items():
                    rationale_positions[labeler][target_dataset][k] = v

    rationale_positions["overlap"] = {}
    for target_dataset in args["new_GT_rationale_datasets"]:
        rationale_positions["overlap"][target_dataset] = {}
        for key in rationale_positions[args["labeler"][0]][target_dataset].keys():
            rationale_positions["overlap"][target_dataset][key] = []
            for each_pos in rationale_positions[args["labeler"][0]][target_dataset][key]:
                if each_pos in rationale_positions[args["labeler"][1]][target_dataset][key]:
                    rationale_positions["overlap"][target_dataset][key].append(each_pos)

    return rationale_positions


from utils_gluex import get_best_model_dict
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Demo of argparse")
    parser.add_argument('--checkpoints_list', nargs='+', help='checkpoints list ')
    parser.add_argument('--device_type', type=str, default="A100", help="device type")
    ARGS = parser.parse_args()
    checkpoints_classify = ARGS.checkpoints_list

    args = {
        'gpu_device': 0,
        'tokenizer': "",
        "new_GT_rationales_dir": "/yanglinyi/new_codes_cluster/RDL/new_GT_rationales",
        "new_GT_rationale_datasets": [
            # "imdb",
            "amazon_polarity",
            "yelp_polarity"
                                      ],
        "labeler": ["1", "2"],
        'train_random_seed': 2022,
        "output_dir": "./feature_score_sst2",
        "max_len": 512,
        "device_type": ARGS.device_type,
        "model_rationale_save_dir": "./model_rationale_sst2",
        "max_target_length": 20,
    }

    from utils_gluex import mkdir

    mkdir(args["output_dir"])
    mkdir(args["model_rationale_save_dir"])
    Original_Text = new_Text_preprocess(args)
    Rationale_Positions = new_GT_preprocess(args)

    batch_size_eval_A100 = {
        "../Models/roberta-base": 1000,
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
        "../Models/roberta-large": 100,
        "../Models/xlnet-large-cased": 100,
        "../Models/xlnet-base-cased": 200,

        "../Models/distilbert-base-uncased":500,
        "../Models/albert-base-v2": 500 ,
        "../Models/bert-base-uncased": 200,
        "../Models/bert-large-uncased":100,
        "../Models/electra-small-discriminator":500,
        "../Models/electra-base-discriminator":300,
        "../Models/electra-large-discriminator":100,

        "../Models/t5-small": 200,
        "../Models/t5-base": 200,
        "../Models/t5-large": 100,

        "../Models/gpt2": 200,
        "../Models/gpt2-medium": 200,
        "../Models/gpt2-large": 100,
        "../Models/bart-base": 200,
        "../Models/bart-large": 100,
    }
    if args["device_type"] == "A100":
        batch_size_eval = batch_size_eval_A100
    elif args["device_type"] == "V100":
        batch_size_eval = batch_size_eval_V100
    else:
        raise ValueError("device type ont supported!")

    best_model_dirs={}
    for checkpoint in checkpoints_classify:
        checkpoint = os.path.basename(checkpoint)
        best_model_dirs[checkpoint] = get_best_model_dict(os.path.join("../evaluation/evaluation_results", checkpoint, checkpoint+"_updated_by_more_tasks.json"))
    print(best_model_dirs)
    for checkpoint in checkpoints_classify:
        checkpoint = os.path.basename(checkpoint)

        for target_dataset in args["new_GT_rationale_datasets"]:
            args["checkpoint"] = checkpoint
            feature_score(best_model_dirs[checkpoint]["sst2"], target_dataset, args, Original_Text, Rationale_Positions)








