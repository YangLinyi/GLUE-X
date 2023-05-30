import os
import numpy as np
import json
import torch

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def calcuate_accu(big_idx, targets):

    n_correct = (big_idx==targets).sum().item()
    return n_correct

def calcuate_accu_for_list(big_idx, targets, args):
    n_correct = ( torch.tensor(big_idx)==torch.tensor(targets) ).sum().item()
    length = len(targets)
    if length==0:
        return 0
    if length < (args["num_partitions"]/2):
        return 0
    return n_correct/length

def calcuate_accu_for_str_list(output_sequences, targets):
    n_correct = 0
    for idx in range(len(targets)):
        if output_sequences[idx]==targets[idx]:
            n_correct += 1
    return n_correct

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def map_seq_to_value(str_list, task_name):
    def convert_sst2(string):
        if string == "positive":
            return 1
        elif string=="negative":
            return 0
        else:
            return -2
    def convert_cola(string):
        if string == "acceptable":
            return 1
        elif string =="unacceptable":
            return 0
        else:
            return -2
    def convert_mrpc(string):
        if string == "equivalent":
            return 1
        elif string =="not_equivalent":
            return 0
        else:
            return -2
    def convert_stsb(string):
        if is_number(string) and float(string)>=0 :
            return float(string)
        else:
            return -2
    def convert_qqp(string):
        if string == "duplicate":
            return 1
        elif string =="not_duplicate":
            return 0
        else:
            return -2
    def convert_mnli(string):
        if string == "entailment":
            return 0
        elif string =="neutral":
            return 1
        elif string =="contradiction":
            return 2
        else:
            return -2
    def convert_qnli(string):
        if string == "entailment":
            return 0
        elif string =="not_entailment":
            return 1
        else:
            return -2
    def convert_rte(string):
        if string == "entailment":
            return 0
        elif string =="not_entailment":
            return 1
        else:
            return -2
    def convert_wnli(string):
        if string == "entailment":
            return 1
        elif string =="not_entailment":
            return 0
        else:
            return -2
    if task_name == "sst2":
        str_list = [convert_sst2(string) for string in str_list]
    elif task_name == "cola":
        str_list = [convert_cola(string) for string in str_list]
    elif task_name == "mrpc":
        str_list = [convert_mrpc(string) for string in str_list]
    elif task_name == "stsb":
        str_list = [convert_stsb(string) for string in str_list]
    elif task_name == "qqp":
        str_list = [convert_qqp(string) for string in str_list]
    elif task_name == "mnli":
        str_list = [convert_mnli(string) for string in str_list]
    elif task_name == "qnli":
        str_list = [convert_qnli(string) for string in str_list]
    elif task_name == "rte":
        str_list = [convert_rte(string) for string in str_list]
    elif task_name == "wnli":
        str_list = [convert_wnli(string) for string in str_list]
    else:
        raise ValueError("task_name not specified in def:map_seq_to_value")

    return torch.tensor(str_list)


def filter_nan_for_dict(results):
    position = list(~np.isnan(np.array(list(results.values()))))
    new_keys = list(np.array(list(results.keys()))[position])
    results = {k: results[k] for k in new_keys}

    return results

def get_best_model_dict(json_path):#for the initial old version code, which also works for the updated version code
    def operate(v):
        if isinstance(v, dict): return v["best_model_dir"]
        else: return v

    with open(json_path) as f:
        temp = json.load(f)
        # print(temp)
        best_models_dirs = {k:operate(v) for k,v in temp.items()}
        # print(best_models_dirs)
    # print(best_models_dirs)
        if "total_cost_time" in best_models_dirs.keys():
            best_models_dirs.pop("total_cost_time")
        # print(best_models_dirs)
    wrong_tasks = []
    for task in best_models_dirs:
        # print(task)
        flag = False
        best_models_dirs[task] = best_models_dirs[task].replace("./", "/yanglinyi/new_codes_cluster/")
        for pth in os.listdir(os.path.dirname(best_models_dirs[task])):
            # print(pth)
            if pth.endswith(".pth"):
                # print(pth)
                if os.path.basename(best_models_dirs[task]) == pth:
                    # print("66666")
                    flag = True
                    break;
        if not flag:
            wrong_tasks.append(task)
    # print(wrong_tasks)

    for task in wrong_tasks:
        # print(task)
        if task == "qnli":
            training_logs_path = os.path.dirname(best_models_dirs[task]) + "/training_logs_reduced_qnli.json"
        elif task == "cola":
            training_logs_path = os.path.dirname(best_models_dirs[task]) + "/training_logs_reduced_cola.json"
        else:
            training_logs_path = os.path.dirname(best_models_dirs[task]) + "/training_logs.json"
        best_models_dir_idx = os.path.basename(best_models_dirs[task]).split(".pt")[0]
        with open(training_logs_path) as f:
            temp = json.load(f)
            validatin_for_each_epoch = temp["validation_for_each_epoch"]
            best_model_result = validatin_for_each_epoch[best_models_dir_idx]
            # print(best_model_result)
        for pth in os.listdir(os.path.dirname(best_models_dirs[task])):#existing_pths
            if pth.endswith(".pth"):
                idx = pth.split(".pt")[0]
                if validatin_for_each_epoch[idx] == best_model_result:
                    best_models_dirs[task] = os.path.dirname(best_models_dirs[task]) + "/" + idx + ".pth"
                    # return best_models_dirs

        # raise ValueError("no existing corresponding best_model_dir")

    return best_models_dirs
# test = get_best_model_dict("./model_performance_full/t5-small/t5-small_reconcatenate.json")
# print(test)

def delete_all_pths_except_last_and_best(root_path, except_pth):
    pths_list = os.listdir(root_path)
    # pths_list.sort(key=lambda x: int(x[:-4]))
    print(pths_list)
    remove_list = []
    for item in pths_list:
        if item.endswith(".pth"):
            continue
        # pths_list.remove(item)
        remove_list.append(item)

    for item in remove_list:
        pths_list.remove(item)

    last_pth_idx = len(pths_list)-1
    print(pths_list)

    for pth in pths_list:
        if pth == os.path.basename(except_pth):
            continue
        elif int(pth[:-4]) == last_pth_idx:
            continue
        else:
            os.remove( os.path.join(root_path, pth) )

def delete_all_pths_except_best(args, root_path, except_pth):
    pths_list = os.listdir(root_path)
    # pths_list.sort(key=lambda x: int(x[:-4]))
    # print(pths_list)

    for item in pths_list:
        if item.endswith(".pth"):
            continue
        pths_list.remove(item)
    # last_pth_idx = len(pths_list)-1
    # print(pths_list)

    for pth in pths_list:
        if pth == os.path.basename(except_pth):
            continue
        # elif int(pth[:-4]) == last_pth_idx:
        #     continue
        else:
            # lr = pth.split("_")[1]
            # batch_size = pth.split("_")[-1].split(".pt")[0]
            # if lr == str(args["lr"]) and batch_size == str(args["per_device_train_batch_size"]):
            #     os.remove( os.path.join(root_path, pth) )
            suffix_except = os.path.basename(except_pth).split("_", 1)[1]
            print(suffix_except)
            # print(except_pth)
            suffix_current = pth.split("_", 1)[1]
            print(suffix_current)
            # print(pth)

            if suffix_current == suffix_except:
                print("successfully deleted")
                os.remove(os.path.join(root_path, pth))

def delete_all_pths_except_best_final(args, root_path, except_pth):
    pths_list = os.listdir(root_path)
    # pths_list.sort(key=lambda x: int(x[:-4]))
    # print(pths_list)

    for item in pths_list:
        if item.endswith(".pth"):
            continue
        pths_list.remove(item)
    # last_pth_idx = len(pths_list)-1
    # print(pths_list)

    for pth in pths_list:
        if pth == os.path.basename(except_pth):
            continue
        # elif int(pth[:-4]) == last_pth_idx:
        #     continue
        else:
            # lr = pth.split("_")[1]
            # batch_size = pth.split("_")[-1].split(".pt")[0]
            # if lr == str(args["lr"]) and batch_size == str(args["per_device_train_batch_size"]):
            #     os.remove( os.path.join(root_path, pth) )
            suffix_except = os.path.basename(except_pth).split("_")[-1]
            print(suffix_except)
            # print(except_pth)
            suffix_current = pth.split("_")[-1]
            print(suffix_current)
            # print(pth)

            if suffix_current == suffix_except:
                print("successfully deleted")
                os.remove(os.path.join(root_path, pth))
def get_task_score(result_dict):
    sum = 0
    num = 0
    for k,v in result_dict.items():
        if k=="best_model_dir":
            continue
        sum += v
        num += 1
    task_score = sum / num
    return task_score

def get_best_hypers(scores):
    max_score = -1
    best_lr = -1
    best_batch_size = -1
    for lr,v0 in scores.items():
        for batch_size, v1 in v0.items():
            if scores[lr][batch_size] > max_score:
                max_score = scores[lr][batch_size]
                best_lr = lr
                best_batch_size = batch_size
    return best_lr, best_batch_size

def mode(test):
    if test:
        return "test_mode"
    else:
        return "full_mode"
# delete_all_pths_except_last_and_best("./results_full/mnli/albert-base-v2", "/results_full/mnli/albert-base-v2/1.pth")