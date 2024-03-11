import os
import csv
import json
import time
import torch
import string
import random
import requests
import argparse
import numpy as np
from random import seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AsyncOpenAI
import asyncio

seed(1234)
np.random.seed(1234)


letters = string.ascii_lowercase

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="data")
parser.add_argument("--model_path", type=str, default="gpt-3.5-turbo")
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--task", type=str, default="EA")
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--iter_num", type=int, default=5)
parser.add_argument("--cot", action="store_true", default=False)
args = parser.parse_args()

task = args.task
lang = args.lang
iter_num = args.iter_num
data_name = args.data_name
model_path = args.model_path
model_name = model_path.split("/")[-1]
letters = string.ascii_lowercase
val_dict = json.load(open("data/dicts.json", "r"))
prompt = val_dict["Prompts"]
columns = val_dict["column_names"]
device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")

if model_name == "random":
    pass
elif model_name in ["gpt-3.5-turbo", "gpt-4"]:
    # Change to your own API Key
    client = AsyncOpenAI(api_key=open("", "r").read())
    client.base_url = ""  # Change to your own base_url
elif model_name in ["Baichuan2-53B", "chatglm3-66b"]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "",  # Change to your own API Key
    }
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    if model_name in ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Yi-6B-Chat"]:
        special_toks = tokenizer.special_tokens_map
        for t in ["sep", "cls", "mask", "pad"]:
            n = f"{t}_token"
            if n not in special_toks:
                tokenizer.add_special_tokens({n: tokenizer.eos_token})
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        .eval()
        .to(device)
    )
    print("Model loaded")


def write_res(d):
    ver = "" if not "permute" in data_name else data_name[-1]
    cot = "_cot" if args.cot else ""
    file_dir = f"data/{task}/Results/{model_name}"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = f"{file_dir}/{model_name}-{lang}{cot}{ver}.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(columns[task])
    with open(file_name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(d)


def get_output(pt, choices=[]):
    llama_based = model_name in [
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Yi-6B-Chat",
    ]
    system_pt = prompt["System" + ("_cot" if args.cot else "")][lang]

    messages = [
        {
            "role": "system",
            "content": system_pt,
        },
        {
            "role": "user",
            "content": pt,
        },
    ]

    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        chat = asyncio.run(
            client.chat.completions.create(
                model=model_path,
                messages=messages,
            )
        )
        pred = chat.choices[0].message.content

    elif model_name == "Baichuan2-53B":
        payload = {"model": "Baichuan2-53B", "messages": messages, "stream": False}
        response = requests.post(
            "https://api.baichuan-ai.com/v1/chat/completions",
            data=json.dumps(payload),
            headers=headers,
            timeout=60,
        )
        if response.status_code == 200:
            pred = eval(response.text)["choices"][0]["message"]["content"]
        else:
            print("请求失败，状态码:", response.status_code)
            print("请求失败，body:", response.text)
            print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
            time.sleep(3)
    elif model_name in ["chatglm3-6b", "Qwen-7B-Chat", "Qwen-14B-Chat"]:
        pred, _ = model.chat(tokenizer, system_pt + pt, history=None)
    elif model_name in ["Baichuan2-7B-Chat", "Baichuan2-13B-Chat"]:
        pred = model.chat(tokenizer, messages)
    elif llama_based:
        chat_template = open("data/llama-2-chat.jinja").read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            chat_template=chat_template,
        ).to(device)
        gen = model.generate(
            input_ids,
            max_new_tokens=500 if args.cot else 75,
            top_p=0.9,
            temperature=0.6,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred = tokenizer.decode(gen[0][input_ids.shape[1] :], skip_special_tokens=True)
        del input_ids
    elif model_name == "random":
        pred = random.sample(choices, 1)[0]
    else:
        print("Model not supported")
    torch.cuda.empty_cache()

    return pred.strip()


def process_output(pred, choices):
    try:
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and " if lang == "en" else "和")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[len(lines) - 1 - j]
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4 if task == "EA" else 6)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4 if task == "EA" else 6)],
                    "option": [
                        f"option {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "choice": [
                        f"choice {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "choice1": [
                        f"choice ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "选项": [
                        f"选项 {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "选项1": [
                        f"选项 ({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                }

                for v in alphabets.values():
                    for a in v:
                        if a in output:
                            return v.index(a)
                for c in choices:
                    if c.lower() in output:
                        return choices.index(c)
                if len(output) == 1 and output in letters[: 4 if task == "EA" else 6]:
                    return letters.index(output)
                if output[0] in letters[: 4 if task == "EA" else 6] and output[1] in [
                    "<",
                    "[",
                    "(",
                    ")",
                    ":",
                ]:
                    return letters.index(output[0])
    except Exception as e:
        print("Error in processing output", type(e).__name__, "–", e)

    return -1


def EU_test(data):
    for i, d in enumerate(tqdm(data)):
        scene, s = [d[t][lang] for t in ["Scenario", "Subject"]]
        e_c, c_c = [d[t]["Choices"][lang] for t in ["Emotion", "Cause"]]
        e_l, c_l = [d[t]["Label"][lang] for t in ["Emotion", "Cause"]]
        # Find label index
        e_li = e_c.index(e_l)
        c_li = c_c.index(c_l)

        e_str = "\n".join([f"({letters[j]}) {c.strip()}" for j, c in enumerate(e_c)])
        c_str = "\n".join([f"({letters[j]}) {c.strip()}" for j, c in enumerate(c_c)])

        e_pt = prompt[task]["Emotion"][lang].format(
            scenario=scene, subject=s, choices=e_str
        ) + (prompt["cot" if args.cot else "no_cot"][lang])

        row = [e_c, e_li, c_c, c_li]
        for _ in range(iter_num):
            try:
                e_response = get_output(e_pt, e_c)
                e_r = process_output(e_response, e_c)
                row += [e_response, e_r]
            except Exception as e:
                print("An error occurred:", type(e).__name__, "–", e)
                row += ["", -1]
                e_r = -1
            try:
                c_pt = prompt[task]["Cause"][lang].format(
                    scenario=scene, emotions=e_c[e_r], subject=s, choices=c_str
                ) + (prompt["cot" if args.cot else "no_cot"][lang])
                c_response = get_output(c_pt, c_c)
                c_r = process_output(c_response, c_c)
                row += [c_response, c_r]
            except Exception as e:
                print("An error occurred:", type(e).__name__, "–", e)
                row += ["", -1]

        write_res(row)


def EA_test(data):
    for i, d in enumerate(tqdm(data)):
        scenario, s, choices, q = [
            d[t][lang] for t in ["Scenario", "Subject", "Choices", "Question"]
        ]
        label = d["Label"]
        c_str = "\n".join(
            [f"({letters[j]}) {c.strip()}" for j, c in enumerate(choices)]
        )
        pt = prompt[task][lang].format(
            scenario=scenario, subject=s, q_type=q, choices=c_str
        ) + (prompt["cot" if args.cot else "no_cot"][lang])

        row = [choices, label]
        for _ in range(iter_num):
            try:
                response = get_output(pt, choices)
                r = process_output(response, choices)
                row += [response, r]
            except Exception as e:
                print("An error occurred:", type(e).__name__, "–", e)
                row += ["", -1]

        write_res(row)


if __name__ == "__main__":
    data = json.loads(open(f"data/{task}/{data_name}.json", "r").read())
    print("+------------------+")
    print(
        f"Running <{model_name}> on <{task}/{data_name}({lang})> [{'with' if args.cot else 'without'} CoT]"
    )
    print("+------------------+")
    if task == "EU":
        EU_test(data)
    elif task == "EA":
        EA_test(data)
    else:
        print("Task not defined")
