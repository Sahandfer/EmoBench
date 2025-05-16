import os
import re
import time
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_yaml, get_response_format, parse_json_response, get_model_name

load_dotenv("./.env")


class LLM:
    def __init__(
        self,
        model_type="openai",
        model_path="gpt-4o",
        num_retries=5,
        device=-1,
        use_cot=False,
        eval_only=False,
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = get_model_name(model_path)
        self.device = device if device >= 0 else "cpu"
        self.use_cot = use_cot
        self.num_retries = num_retries
        self.prompts = load_yaml("src/configs/prompts.yaml")

        if not eval_only:
            self.load_model()

    def init_prompt(self, task, lang):
        self.task = task
        self.lang = lang
        sys_prompt = self.prompts["sys"][lang]
        output_format = get_response_format(task, lang, use_cot=self.use_cot)
        sys_prompt += output_format
        if "qwen" in self.model_name.lower() and not self.use_cot:
            sys_prompt = "/no_think\n\n" + sys_prompt
        self.messages = [{"role": "system", "content": sys_prompt}]

    def load_model(self):
        if self.model_type == "openai":
            self.client = ChatOpenAI(
                model=self.model_path,
                api_key=os.environ.get("API_KEY"),
                temperature=0.6,
            )
        elif self.model_type == "openai-compatible":
            self.client = ChatOpenAI(
                model=self.model_path,
                base_url=os.environ.get("API_URL"),
                api_key=os.environ.get("API_KEY"),
                temperature=0.6,
            )
        elif self.model_type == "HF":
            self.client = pipeline(
                "text-generation", model=self.model_path, device=self.device
            )
        else:
            print("Model is not supported at the moment")
            self.client = None

        print(f"> {self.model_name} loaded successfully")

    def gen_response(self, msg):
        messages = self.messages + [{"role": "user", "content": msg}]
        for i in range(self.num_retries):
            try:
                if self.model_type == "HF":
                    response = self.client(
                        messages,
                        max_new_tokens=2048 if self.use_cot else 50,
                        return_full_text=False,
                    )[0]
                    response = response["generated_text"]
                else:
                    response = self.client.invoke(messages)
                    response = response.content
                response = parse_json_response(response)
                if response is None:
                    raise ValueError("No JSON response found")
                return response
            except Exception as e:
                print(f"Error: {e}")
                # Sleep for API calls
                if self.model_type != "HF":
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                return (
                    {"answer": ""}
                    if self.task == "EA"
                    else {
                        "answer_q1": "",
                        "answer_q2": "",
                    }
                )
