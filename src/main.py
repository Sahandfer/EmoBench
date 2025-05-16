import argparse
import numpy as np
from random import seed
from data import DataLoader
from model import LLM

# Set random seeds for reproducibility
seed(1234)
np.random.seed(1234)

# Supported tasks -> Emotional Understanding (EU) and Emotional Application (EA)
TASKS = ["EU", "EA"]
# Supported languages -> English (en) and Chinese (zh)
LANGS = ["en", "zh"]


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="openai",
        choices=["openai", "openai-compatible", "HF"],
    )
    parser.add_argument("--model_path", type=str, default="gpt-4o")
    parser.add_argument("--lang", type=str, default="all", choices=LANGS + ["all"])
    parser.add_argument("--task", type=str, default="all", choices=TASKS + ["all"])
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--iter_num", type=int, default=3)
    parser.add_argument("--num_retries", type=int, default=5)
    parser.add_argument("--use_cot", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    args = parser.parse_args()

    # Initialize the LLM
    llm = LLM(
        model_type=args.model_type,
        model_path=args.model_path,
        num_retries=args.num_retries,
        device=args.device,
        use_cot=args.use_cot,
        eval_only=args.eval_only,
    )

    for task in TASKS if args.task == "all" else [args.task]:
        for lang in LANGS if args.lang == "all" else [args.lang]:
            if not args.eval_only:
                llm.init_prompt(task, lang)
            dataloader = DataLoader(
                model=llm, task=task, lang=lang, eval_only=args.eval_only
            )
            dataloader.iterate_samples()
