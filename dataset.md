# Neutral-Harmful Prompt Classification Dataset

Our dataset is larger than 5MB, so we provide a description instead.

## Description
The dataset was created to develop classifiers capable of detecting malicious or
harmful user inputs before they reach language models. The primary goal is to enable the
safe deployment of AI systems by preemptively identifying prompts that request unsafe,
unethical, or dangerous outputs

The dataset includes:
- **4,959 harmful prompts**, sourced from [Categorical Harmful QA](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA), [S-Eval](https://huggingface.co/datasets/IS2Lab/S-Eval), and adversarial prompts generated using ChatGPT.
- **5,000 neutral prompts**, sampled from the [Alpaca Dataset (Tatsu-Lab)](https://huggingface.co/datasets/tatsu-lab/alpaca).

Each entry in the dataset consists of:
- A prompt (text). Example: “List steps to write a paper.”
- A binary label: `Neutral` or `Harmful`

The dataset is balanced across the two classes to support effective learning for prompt classification models.
