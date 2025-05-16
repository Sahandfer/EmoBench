import os
import json
import yaml
import string
import pandas as pd
from utils import load_yaml, save_gen_results, save_eval_results
from tqdm.auto import tqdm

letters = string.ascii_uppercase


class DataLoader:
    def __init__(
        self, model=None, task="EU", lang="en", data_path="data", eval_only=False
    ):
        self.client = model
        self.lang = lang
        self.task = task
        prompts = load_yaml("src/configs/prompts.yaml")
        self.prompt = prompts[self.task][self.lang]
        self.data_path = f"{data_path}/{task}.jsonl"
        self.eval_only = eval_only

        if not eval_only:
            self.load_samples()

    def load_samples(self):
        data = pd.read_json(path_or_buf=self.data_path, lines=True, encoding="utf-8")
        self.data = data[data["language"] == self.lang]
        print(f"> Loaded {self.task} ({self.lang}) with {len(self.data)} samples")

        file_path = f"results/{self.task}/{self.client.model_name}.jsonl"
        if os.path.exists(file_path):
            data = pd.read_json(path_or_buf=file_path, lines=True, encoding="utf-8")
            data = data[data["lang"] == self.lang]
            print(f"> Found checkpoint with {len(data)}/200 samples")
            self.data = self.data[~self.data["qid"].isin(data["qid"])]

    def load_eval_results(self):
        file_path = f"results/{self.task}/{self.client.model_name}.jsonl"
        if not os.path.exists(file_path):
            print("No results found! Please run generation first.")
        else:
            responses = pd.read_json(
                path_or_buf=file_path, lines=True, encoding="utf-8"
            )
            responses = responses[responses["lang"] == self.lang]
            if len(responses) != 200:
                print(
                    f"> Results are not complete (n = {len(responses)}/200), you should run the model again for missing samples"
                )
            return responses

    def rank_choices(self, choices):
        output = []
        for i, c in enumerate(choices):
            output.append(f"{letters[i]}) {c}")
        return "\n".join(output)

    def iterate_samples(self):
        if not self.eval_only and len(self.data):
            self.generate_responses()
        self.evaluate_results()

    def generate_responses(self):
        print(
            f"> Generating {self.task}-{self.lang} responses for {self.client.model_name}..."
        )

        for idx, sample in tqdm(self.data.iterrows()):
            if self.task == "EU":
                msg = self.prompt.format(
                    scenario=sample["scenario"],
                    subject=sample["subject"],
                    emo_choices=self.rank_choices(sample["emotion_choices"]),
                    cause_choices=self.rank_choices(sample["cause_choices"]),
                )
                response = self.client.gen_response(msg)
                res = {
                    "qid": sample["qid"],
                    "lang": sample["language"],
                    "coarse_category": sample["coarse_category"],
                    "finegrained_category": sample["finegrained_category"],
                    "emo_label": letters[
                        sample["emotion_choices"].index(sample["emotion_label"])
                    ],
                    "emo_answer": response.get("answer_q1", ""),
                    "cause_label": letters[
                        sample["cause_choices"].index(sample["cause_label"])
                    ],
                    "cause_answer": response.get("answer_q2", ""),
                }

            elif self.task == "EA":
                msg = self.prompt.format(
                    scenario=sample["scenario"],
                    subject=sample["subject"],
                    choices=self.rank_choices(sample["choices"]),
                    q_type=sample["question type"],
                )
                response = self.client.gen_response(msg)
                res = {
                    "qid": sample["qid"],
                    "lang": sample["language"],
                    "category": sample["category"],
                    "label": letters[sample["choices"].index(sample["label"])],
                    "answer": response.get("answer", ""),
                }

            save_gen_results(res, self.task, self.client.model_name)

    def evaluate_results(self):
        print(
            f"> Evaluating {self.task}-{self.lang} results for {self.client.model_name}..."
        )
        responses = self.load_eval_results()
        if responses is not None and len(responses):
            if self.task == "EA":
                responses["accuracy"] = responses["label"] == responses["answer"]
                results = responses.groupby("category")["accuracy"].mean().reset_index()
                results = results.set_index("category").to_dict()["accuracy"]
            elif self.task == "EU":
                responses["accuracy"] = (
                    responses["emo_label"] == responses["emo_answer"]
                ) & (responses["cause_label"] == responses["cause_answer"])
                results = (
                    responses.groupby("coarse_category")["accuracy"]
                    .mean()
                    .reset_index()
                )
                results = results.set_index("coarse_category").to_dict()["accuracy"]

            results["Overall"] = responses["accuracy"].mean()

            save_eval_results(results, self.task, self.lang, self.client.model_name)
