import os
import string
import argparse
import collections
import numpy as np
import pandas as pd
from random import seed
from tabulate import tabulate

seed(1234)
np.random.seed(1234)

letters = string.ascii_lowercase

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
parser.add_argument("--task", type=str, default="EA")
parser.add_argument("--iter_num", type=int, default=5)
parser.add_argument("--eval_all", action="store_true", default=False)
args = parser.parse_args()

task = args.task
iter_num = args.iter_num
model_name = args.model_name
eval_all = args.eval_all


def get_pred(row, n):
    pre_preds = [row[f"{n}Pred {i+1}"] for i in range(iter_num)]
    commons = collections.Counter(pre_preds).most_common()
    maj_vote = (
        commons[0][0]
        if commons[0][0] != -1
        else (commons[1][0] if len(commons) > 1 else -1)
    )
    return int(maj_vote)


def print_res(res):
    row = []
    headers = ["English", "Chinese", "English (CoT)", "Chinese (CoT)"]
    types = ["-en", "-zh", "-en_cot", "-zh_cot"]
    if eval_all:
        headers = ["Model"] + headers
        for k, v in res.items():
            row.append([k] + [v.get(k + t, "0") for t in types])
    else:
        row = [[res.get(model_name + k, "0") for k in types]]

    print(tabulate(row, headers=headers))


def evaluate(model_name, task, eval_all=False):
    res = {}
    directory = f"data/{task}/Results/{model_name}"
    res_files = sorted([f for f in os.listdir(directory) if f.endswith(".csv")])

    for f in res_files:
        file_dir = directory + "/" + f
        f = f.replace(".csv", "")
        f = f[:-1] if f[-1] in ["1", "2", "3"] else f
        df = pd.read_csv(file_dir)

        if len(df) == 200:
            if task == "EA":
                df["Pred"] = df.apply(lambda x: get_pred(x, ""), axis=1)
                df["Res"] = df["Pred"] == df["Label"].astype(int)
            else:
                for n in ["Emo ", "Cause "]:
                    df[f"{n}Pred"] = df.apply(lambda x: get_pred(x, n), axis=1)
                    df[f"{n}Res"] = df[f"{n}Pred"] == df[f"{n}Label"].astype(int)
                df["Res"] = df["Emo Res"] & df["Cause Res"]

            acc = (sum(df[f"Res"]) / len(df[f"Res"])) * 100
            res[f] = res.get(f, []) + [acc]
        else:
            print(f"Samples for {file_dir} are not enough.")

    for k, v in res.items():
        res[k] = "{:0.2f}".format(sum(v) / len(v))

    if not eval_all:
        print_res(res)
    else:
        return res


def evaluate_all(task):
    res_all = {}
    directory = f"data/{task}/Results"
    for model_name in os.listdir(directory):
        res_all[model_name] = evaluate(model_name, task, eval_all=True)
    print_res(res_all)


if __name__ == "__main__":
    if eval_all:
        print("+------------------+")
        print(f"Running evaluation for <{task}>")
        print("+------------------+")
        evaluate_all(task)
    else:
        print("+------------------+")
        print(f"Running evaluation on <{model_name}> for <{task}>")
        print("+------------------+")
        evaluate(model_name, task)
