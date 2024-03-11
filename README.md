# EmoBench
> This is the official repository for our paper ["EmoBench: Evaluating the Emotional Intelligence of Large Language Models"](https://arxiv.org/abs/2402.12071)

<img src="https://img.shields.io/badge/Venue-ACL--24-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Under Review-success" alt="status"/> <img src="https://img.shields.io/badge/Contributions-Welcome-red"> <img src="https://img.shields.io/badge/Last%20Updated-2024--03--11-2D333B" alt="update"/>

![EmoBench](EmoBench.jpg)

## Announcements

- :construction: We have made some minor changes to the data to improve the case fluency and choice difficulty. So the provided results may be outdated. We will be updating the results soon. We also plan to evaluate newer LLMs, such as Gemma.
- :star: Star the project to get notified when this repository is updated.
- :tada: Thank you for your interest and support! 

## Usage Guide

### Main Structure
**Generation**: For running a supported model on EmoBench, run the following command:

```sh
python generate.py \
  --data_name [data_name] \
  --model_path [model_path] \
  --lang [en|zh] \
  --task [EA|EU]\
  --device [device] \
  --iter_num [iter_num] \
  [--cot] 
```

`--data_name`: the name of the json file for the dataset. Defaults to "data".

`--model_path`: the path to model weights or name of the model to evaluate. Defaults to "gpt-3.5-turbo".

`--lang`: the language for the evaluation. Defaults to "en" for English. 

`--task`: the task for generation. Defaults to "EA". Choose "EU" for Emotional Understanding.

`--device`: the ID of the GPU to run the model. Defaults to -1 for "CPU".

`--iter_num`: the number of responses to generate per sample. Defaults to 5.

`--cot`: enables chain-of-thought reasoning. It is set to False by default.

*Note*: The chat template for LLama-based inference was adopted from [Chat Templates](https://github.com/chujiezheng/chat_templates).


**Evaluation**: For evaluating the generated responses, simply run the following:

```
python evaluate.py \
  --model_name [model_name] \
  --task [EA|EU]\
  --iter_num [iter_num] \
  [--eval_all]
```

`--model_name`: the name of the model to evaluate (same as the folder name in results). Defaults to "gpt-3.5-turbo".

`--task`: the task for evaluation. Defaults to "EA". Choose "EU" for Emotional Understanding.

`--iter_num`: the number of responses that were generated per sample. Defaults to 5.

`--eval_all`: evaluates all the models regardless of `--model_name`.


### Supported Models
1. Yi-Chat (6B)
2. ChatGLM3 (6B)
3. Llama-2-chat (7B & 13B)
4. Baichuan2 (7B, 13B & 53B)
5. Qwen (7B & 14B)
6. Gemma (7B) --> soon
7. ChatGPT (gpt-3.5-turbo & gpt-4)

### Benchmarking your own model
Simply add the logic for your model's inference in the `get_output` function.

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{EmoBench2024,
     title={EmoBench: Evaluating the Emotional Intelligence of Large Language Models}, 
      author={Sahand Sabour and Siyang Liu and Zheyuan Zhang and June M. Liu and Jinfeng Zhou and Alvionna S. Sunaryo and Juanzi Li and Tatia M. C. Lee and Rada Mihalcea and Minlie Huang},
			year={2024},
      eprint={2402.12071},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
