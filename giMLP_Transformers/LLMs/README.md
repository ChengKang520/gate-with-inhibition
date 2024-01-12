#  Finetune LLMs with InA

> For more finetune methods for LLM, please see [LLM-Finetune-Guide](https://github.com/A-baoYang/LLM-Finetune-Guide)

This repository is a tutorial for finetuning LLMs with InA on Alpaca datasets! 

So here's how to reproduce:

## Installation

1. Install requirements

```bash
$ pip install -r requirements.txt
```

2. Install PyTorch at compatible version with CUDA

```bash
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```


## Datasets

This repository combined all datasets using English-instruction:

1. `alpaca_data.json`: Original dataset from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
2. `alpaca_data_cleansed.json`: Cleansing by [gururise/AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)


## Finetune

Reference finetune method provide by [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) 

1. Run on 1 GPU with Colab: https://colab.research.google.com/drive/1QvtrJpikkkNKSbwwG766SIGbBw2TQRd5?usp=sharing

  - `LLaMA`
    ```bash
    $ cd finetune/
    $ python finetune.py --base_model decapoda-research/llama-7b-hf --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/llama-7b-hf_alpaca-en-zh --threshold_ratio_inhi 0.3 --lora_target_modules '["q_proj", "v_proj"]'
    ```
  
  - `BLOOM`
    ```bash
    $ cd finetune/
    $ python finetune.py --base_model bigscience/bloomz-7b1-mt --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/bloomz-7b1-mt_alpaca-en-zh --threshold_ratio_inhi 0.3 --lora_target_modules '["query_key_value"]'
    ```

2. Use `torchrun` for distributed training on Multi-GPUs

  - `LLaMA`
    ```bash
    $ cd finetune/
    $ torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py --base_model decapoda-research/llama-7b-hf --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/llama-7b-hf_alpaca-en-zh --threshold_ratio_inhi 0.3 --lora_target_modules '["q_proj", "v_proj"]'
    ```

  - `BLOOM`
    ```bash
    $ cd finetune/
    $ torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py --base_model bigscience/bloomz-7b1-mt --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/bloomz-7b1-mt_alpaca-en-zh --threshold_ratio_inhi 0.3 --lora_target_modules '["query_key_value"]'
    ```

![](https://i.imgur.com/Czw3AAx.png)

### Finetune Domain Tasks

I've collected different domain tasks in my repository: [instruction-finetune-datasets](https://github.com/ChengKang520/psychotherapy-assistant_instruction)

Welcome cooperations! Please contact me at: `kangkangsome@gmail.com`. I'd like to try tasks from different domains such as investment, fraud, e-commerce, law, healthcare, ...


## Model Serving
To serve your own model service through API & simple website UI!

1. Model API


    ```bash
    $ cd serve/
    $ python api.py
    ```

2. demo UI


    ```bash
    $ cd serve/
    $ python ui.py
    ```



```
@article{kang4551993ina,
  title={InA: Inhibition Adaption on Pre-Trained Language Models},
  author={Kang, Cheng and Prokop, Jindich and Tong, Lei and Zhou, Huiyu and Hu, Yong and Novak, Daniel},
  journal={Available at SSRN 4551993}
}
```

