import os.path
import nltk
nltk.download('punkt')
import sys
sys.path.append("../evaluation")
import sys
# sys.setdefaultencoding('utf-8')
from tqdm import tqdm_notebook
# import nltk
# nltk.download('punkt')

import numpy as np
np.random.seed(2022)
import random
random.seed(2022)

import torch
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from sklearn.utils import shuffle
import transformers
import json



from datasets import load_metric,load_dataset,Value
import csv


import nltk
nltk.data.path.append('D:\\python_pkg_data\\nltk_data')
# from nltk.tokenize import word_tokenize
from esnli_preprocess import esnli_word_tokenize as word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


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

import argparse
parser = argparse.ArgumentParser("Demo of argparse")
parser.add_argument('--checkpoints_list', nargs='+', help='checkpoints list ')

parser.add_argument('--device_type', type=str, default="A100", help="device type")
parser.add_argument('--by_pieces', action="store_true", help="by pieces")
parser.add_argument('--piece_start_idx', type=int, default = -1, help="device type")
parser.add_argument('--piece_end_idx', type=int, default = -1, help="device type")



ARGS = parser.parse_args()
checkpoints_classify = ARGS.checkpoints_list


args = {
    'gpu_device':0,
    "new_GT_rationale_datasets":["snli"],
    "labeler":["1"],
    'train_random_seed':2022,
    "output_dir":"./feature_score_mnli",
    "max_len":512,
    "device_type":ARGS.device_type,
    "model_rationale_save_dir":"./model_rationale_mnli",
    "max_target_length":20,
    "by_pieces":ARGS.by_pieces,
    "by_pieces_output_dir":"./model_rationale_mnli_by_pieces",
    "piece_start_idx":ARGS.piece_start_idx,
    "piece_end_idx": ARGS.piece_end_idx,

}
from utils_gluex import mkdir
mkdir(args["output_dir"])
mkdir(args["model_rationale_save_dir"])


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

##################for t5 only#####################################
class CustomerDataset_t5(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        if self.labels[idx] == 0:
            item["labels"] = "entailment"
        elif self.labels[idx] == 1:
            item["labels"] = "neutral"
        elif self.labels[idx] == 2:
            item["labels"] = "contradiction"
        # print(item)
        return item

    def __len__(self):
        return len(self.labels)

def map_label_to_idx(label, args):
    return args["tokenizer"](label).input_ids[0]

def get_rationale_spans_t5(model, text, label, args, topk=5,):
    #     text = imdb_texts[example_idx]
    #     label = imdb_labels[example_idx]
    token_text = word_tokenize(text)

    candidates, remove_terms = identify_important_terms(token_text, text)
    #     print(candidates)
    #     print(candidates[-1])
    # print(candidates[0])
    # print(candidates[0].split(": ")[1])
    candidates_label = [label] * len(candidates)

    candidates_encodings = args['tokenizer'](candidates, truncation = True, max_length=args["max_len"], pad_to_max_length=True,)

    candidates_dataset = CustomerDataset_t5(candidates_encodings, candidates_label)
    candidates_dataloader = DataLoader(candidates_dataset, batch_size=batch_size_eval[os.path.join("../Models",args["checkpoint"])], shuffle=False)

    model.eval()
    output_logits = []
    # print("pad_token_id")
    # print(args["tokenizer"].pad_token_id)
    for batch in tqdm_notebook(candidates_dataloader):
        # batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
        batch["input_ids"] = batch["input_ids"].cuda(args["gpu_device"])
        batch["attention_mask"] = batch["attention_mask"].cuda(args["gpu_device"])
        #         print(batch)
        with torch.no_grad():
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            #             outputs = model(**batch)
            targets = batch["labels"]
            targets_encoding = args["tokenizer"](targets,
                                         pad_to_max_length=True,
                                         max_length=args["max_target_length"],
                                         truncation=True,
                                         add_special_tokens=True,
                                         return_tensors="pt"
                                         ).input_ids
            targets_encoding[targets_encoding == args["tokenizer"].pad_token_id] = -100
            # print(targets_encoding[32])
            targets_encoding = targets_encoding.to(args["gpu_device"], dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask,
                            labels=targets_encoding
                            )

            logits = outputs.logits
            # logits = logits.softmax(-1)
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
############################################################33
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


def get_rationale_spans(model, text, label, args, topk=5,):
    #     text = imdb_texts[example_idx]
    #     label = imdb_labels[example_idx]
    token_text = word_tokenize(text)

    candidates, remove_terms = identify_important_terms(token_text, text)
    #     print(candidates)
    #     print(candidates[-1])

    candidates_label = [label] * len(candidates)

    candidates_encodings = args['tokenizer'](candidates, truncation=True, max_length=args["max_len"], pad_to_max_length=True,)

    candidates_dataset = CustomerDataset(candidates_encodings, candidates_label)
    candidates_dataloader = DataLoader(candidates_dataset, batch_size=batch_size_eval[os.path.join("../Models",args["checkpoint"])], shuffle=False)

    model.eval()
    output_logits = []
    for batch in tqdm_notebook(candidates_dataloader):
        batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
        #         print(batch)
        with torch.no_grad():
            #             outputs = model(**batch)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        #             print(logits)
        #             return

        #         logits = outputs.logits
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
        #         print(ids, remove_terms[ids]['terms'])
        span = [i for i in range(remove_terms[ids]['start_token'], remove_terms[ids]['end_token'])]
        inferred_spans.append(span)

    inferred_pos = []
    for span in inferred_spans:
        for number in span:
            inferred_pos.append(number)

    inferred_pos = list(set(inferred_pos))

    return inferred_pos

from esnli_preprocess import remove_punc
def metrics(pos, token_text, pos_GT, mode):
    if mode == "num":
        correct = 0
        print(pos)
        print(pos_GT)
        for p in pos:
            if p in pos_GT:
                correct += 1
    elif mode == "str":
        pos = [token_text[each_pos] for each_pos in pos]
        pos = remove_punc(pos)
        pos = list(filter(bool, pos))
        common = []
        print(pos)
        print(pos_GT)
        for s1 in pos:
            for s2 in pos_GT:
                if s1 == s2:
                    common.append(s1)
        correct = len(common)
    else:
        raise ValueError("Metric mode not specified")

    print(correct)
    if correct == 0:
        return 0, 0, 0

    precision = correct / len(pos)
    recall = correct / len(pos_GT)

    f1 = 2 * precision * recall / (precision + recall)
    # print(f1)
    # print(precision)
    # print(recall)

    return f1, precision, recall


def feature_score(model_dir, target_dataset, args):

    model = torch.load(model_dir).module.cuda(args['gpu_device'])
    average_f1 = {}
    average_precision = {}
    average_recall = {}
    for labeler in Rationale_Positions.keys():
        average_f1[labeler] = 0
        average_precision[labeler] = 0
        average_recall[labeler] = 0

    original_text = Original_Text[target_dataset]
    if args["by_pieces"]:
        model_rationale_save_path = os.path.join(args["model_rationale_save_dir"],
                                                 "{0}_to_{1}".format(
                                                     str(args["piece_start_idx"]), str(args["piece_end_idx"])),
                                                 args["checkpoint"] + "_rationale.csv")
    else:
        model_rationale_save_path = os.path.join(args["model_rationale_save_dir"],
                                                 args["checkpoint"] + "_rationale.csv")
    mkdir(os.path.dirname(model_rationale_save_path))
    with open(model_rationale_save_path, 'w',
              newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text",
                         "model_rationale", "model_rationale_pos",
                         "human_rationale", "human_rationale_pos"])

    for key in list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()):
        # print(key)
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
            f1, precision, recall = metrics(pos=generated_rationales_spans, token_text=token_text, pos_GT=Rationale_Positions[labeler][target_dataset][key], mode="num")
            average_f1[labeler] += f1
            average_recall[labeler] += recall
            average_precision[labeler] += precision

        with open(model_rationale_save_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            sorted_generated_rationales_spans = sorted(generated_rationales_spans)
            sorted_human_rationale_pos= sorted(Rationale_Positions[labeler][target_dataset][key])
            model_rationale = [token_text[each_pos] for each_pos in sorted_generated_rationales_spans]
            human_rationale = [token_text[each_pos] for each_pos in sorted_human_rationale_pos]
            writer.writerow([original_text[key],
                             model_rationale, sorted_generated_rationales_spans,
                             human_rationale, sorted_human_rationale_pos
                             ])
    for labeler in Rationale_Positions.keys():
        average_f1[labeler] = average_f1[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))
        average_precision[labeler] = average_precision[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))
        average_recall[labeler] = average_recall[labeler] / len(list(Rationale_Positions[args["labeler"][0]][target_dataset].keys()))

    if args["by_pieces"]:
        score_save_path = os.path.join(args["output_dir"],
                                       "{0}_to_{1}".format(str(args["piece_start_idx"]), str(args["piece_end_idx"])),
                                       args["checkpoint"], target_dataset+".json")
    else:
        score_save_path = os.path.join(args["output_dir"], args["checkpoint"], target_dataset+".json")
    mkdir(os.path.dirname(score_save_path))
    with open(score_save_path, "w") as f:
        f.write(json.dumps({
            "dataset_name":target_dataset,
            "f1_score":average_f1,
            "precision":average_precision,
            "recall":average_recall,
            "dataset_length":len(list( Rationale_Positions[args["labeler"][0]][target_dataset].keys() )),
        }
            , ensure_ascii=False, indent=4, separators=(',', ':')) + '\n')



from utils_gluex import get_best_model_dict

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
best_model_dirs={}
for checkpoint in checkpoints_classify:
    checkpoint = os.path.basename(checkpoint)
    best_model_dirs[checkpoint] = get_best_model_dict(
        os.path.join("../evaluation/evaluation_results", checkpoint, checkpoint + "_updated_by_more_tasks.json"))

print(best_model_dirs)

from esnli_preprocess import get_snli_dataset

for checkpoint in checkpoints_classify:
    checkpoint = os.path.basename(checkpoint)
    if "t5" in checkpoint:
        Rationale_Positions, Original_Text = get_snli_dataset(mode="t5")
    else:
        Rationale_Positions, Original_Text = get_snli_dataset(mode="classifier")
    if args["by_pieces"]:
        for labler in Rationale_Positions.keys():
            for target_dataset in Rationale_Positions[labler].keys():
                dataset_key_list = list(Rationale_Positions[labler][target_dataset].keys())
                sliced_idx_list = list(range(args["piece_start_idx"], args["piece_end_idx"]))
                selected_key_list = [dataset_key_list[selected_idx] for selected_idx in sliced_idx_list]
                temp = {each_selected_key:Rationale_Positions[labler][target_dataset][each_selected_key]
                        for each_selected_key in selected_key_list}
                Rationale_Positions[labler][target_dataset] = temp
        for target_dataset in Original_Text.keys():
            dataset_key_list = list(Original_Text[target_dataset].keys())
            sliced_idx_list = list(range(args["piece_start_idx"], args["piece_end_idx"]))
            selected_key_list = [dataset_key_list[selected_idx] for selected_idx in sliced_idx_list]
            temp = {each_selected_key: Original_Text[target_dataset][each_selected_key]
                    for each_selected_key in selected_key_list}
            Original_Text[target_dataset] = temp
    for target_dataset in args["new_GT_rationale_datasets"]:
        args["checkpoint"] = checkpoint
        feature_score(best_model_dirs[checkpoint]["mnli"], target_dataset, args)







