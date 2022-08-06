<img src="../figures/palm.gif" width="450px"></img>
# giMLP on Transformers On Downstream Language Tasks Fine-Tuning 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

This is a PyTorch implementation of the paper [giMLPs: Gate with Inhibition Mechanism in MLPs]().


## Updates
- (05/08/2022) Initial release.



# Try Gate With Inhbition When Fine-Tuning



## Get Started

### Environment
Install Python 3.9, PyTorch 1.9.0 and packages ``` pip install -r requirements.txt ```.

### Experimental Data 
You can obtain all the six benchmarks from Tsinghua Cloud or Google Drive. All the datasets are well pre-processed and can be used easily.
- GLUE. The General Language Understanding Evaluation (GLUE) benchmark is a collection of nine natural language understanding (NLU) tasks: single-sentence tasks CoLA and SST-2, similarity and paraphrasing tasks MRPC, STS-B and QQP, and natural language inference tasks MNLI, QNLI, RTE and WNLI.

- SQuAD v1.1/v2.0 is the Stanford Question Answering Dataset (SQuAD) v1.1 and v2.0, two popular machine reading comprehension benchmarks from approximately 500Wikipedia articles with questions and answers obtained by crowdsourcing. The SQuAD v2.0 dataset includes unanswerable questions about the same paragraphs.

- SWAG is a large-scale adversarial dataset for the task of grounded commonsense inference, which unifies natural language inference and physically grounded reasoning. SWAG consists of 113k multiple choice questions about grounded situations.

- CoNLL 2003 is an English dataset consisting of text from a wide variety of sources. It has 4 types of named entities.

### Pre-trained Models Used Here 

|Model        | Vocabulary Size|Backbone Parameters(M)| Hidden Size | Layers|  Attention Heads|
|-------------|------|---------|-----|-----|---------|
|[BERT-Large](https://huggingface.co/bert-large-uncased)|30522|336|1024| 24| 16|
|[RoBERTa-Large](https://huggingface.co/roberta-large)|50265|355|1024| 24| 16|
|[DeBERTa-V2-Large](https://huggingface.co/microsoft/deberta-large)|128100|304|1024| 24| 16|
|[DeBERTa-V3-Large](https://huggingface.co/microsoft/deberta-v3-large)|128100|304|1024| 24| 16|

 



## Insert Gate With Inhibition Into DeBERTaV3


### Change the config.json of DeBERTaV3 Model To

There are several fine-tuning strategies on using Gate With Inhibition mechanism:  
- {"gi_key_side": true,} means inserting gi into attention block's Key side is TRUE; 
- {"gi_query_side": true,} means inserting gi into attention block's Query side is TRUE;
- {"inhibition_level": 0.3,} means that the inhibition level is 30%;
- {"gi_layer_num": 24,} means inserting gi into how many last layers.


``` Python
{
  "_name_or_path": "microsoft/deberta-v3-large",
  "architectures": [
    "DebertaV2ForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": "cola",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "id2label": {
    "0": "unacceptable",
    "1": "acceptable"
  },
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "label2id": {
    "acceptable": 1,
    "unacceptable": 0
  },
  "layer_norm_eps": 1e-07,
  "max_position_embeddings": 512,
  "max_relative_positions": -1,
  "model_type": "deberta-v2",
  "norm_rel_ebd": "layer_norm",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pooler_dropout": 0,
  "pooler_hidden_act": "gelu",
  "pooler_hidden_size": 1024,
  "pos_att_type": [
    "p2c",
    "c2p"
  ],
  "position_biased_input": false,
  "position_buckets": 256,
  "relative_attention": true,
  "share_att_key": true,
  "gi_key_side": true,
  "gi_query_side": true,
  "inhibition_level": 0.3,
  "gi_layer_num": 24,
  "torch_dtype": "float32",
  "transformers_version": "4.19.2",
  "type_vocab_size": 0,
  "vocab_size": 128100
}

```

### The Gate With Inhibition Codes In DeBERTaV3 


``` Python

# To apply gate with inhibition mechanism in DeBERTa, you need to make two changes on your code,
# 1. change the config in the class DisentangledSelfAttention(nn.Module) in script modelling_deberta_v2.py:

    self.threshold_ratio = config.inhibition_level
    self.gi_key_side = config.gi_key_side
    self.gi_query_side = config.gi_query_side
    self.gi_attention_heads_i = self.num_attention_heads
    self.hidden_size = config.hidden_size
    self.layer_i = layer_i
    self.last_layers_num = config.gi_layer_num
    self.num_hidden_layers = config.num_hidden_layers
    if self.layer_i > config.num_hidden_layers - self.last_layers_num:
        if self.gi_key_side:
          self.inhibition_k = nn.Linear(config.hidden_size, self.all_head_size)
          self.select_k = nn.Linear(config.hidden_size, self.all_head_size)
        if self.gi_query_side:
          self.inhibition_q = nn.Linear(config.hidden_size, self.all_head_size)
          self.select_q = nn.Linear(config.hidden_size, self.all_head_size)


# 2. Change the architecture in DeBERta in the class         
"""
      Call the module
      Args:
          hidden_states (`torch.FloatTensor`):
              Input states to the module usually the output from previous layer, it will be the Q,K and V in
              *Attention(Q,K,V)*
          attention_mask (`torch.ByteTensor`):
              An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
              sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
              th token.
          output_attentions (`bool`, optional):
              Whether return the attention matrix.
          query_states (`torch.FloatTensor`, optional):
              The *Q* state in *Attention(Q,K,V)*.
          relative_pos (`torch.LongTensor`):
              The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
              values ranging in [*-max_relative_positions*, *max_relative_positions*].
          rel_embeddings (`torch.FloatTensor`):
              The embedding of relative distances. It's a tensor of shape [\\(2 \\times
              \\text{max_relative_positions}\\), *hidden_size*].
      """
      threshold_ratio = self.threshold_ratio

      if query_states is None:
          query_states = hidden_states

      if self.gi_query_side:
          if self.num_attention_heads!= self.gi_attention_heads_i:
              self.segment_num = self.num_attention_heads - self.gi_attention_heads_i
              query_states_head, query_states_tail = torch.split(query_states, [self.segment_num, self.gi_attention_heads_i], dim=0)
              rehearsal_states_q = nn.functional.gelu(self.inhibition_q(query_states_tail))
              threshold_inhi_q = torch.max(self.select_q(rehearsal_states_q)) * threshold_ratio
              selection_states_q = nn.functional.gelu(self.select_q(rehearsal_states_q) - threshold_inhi_q) + 1
              query_layer_head = self.query_proj(query_states_head)
              query_layer_tail = torch.mul(selection_states_q, self.query_proj(query_states_tail))
              query_layer_temp = torch.cat((query_layer_head, query_layer_tail), 0)
              query_layer = self.transpose_for_scores(query_layer_temp, self.num_attention_heads)
          else:
              if self.layer_i > self.num_hidden_layers - self.last_layers_num:
                rehearsal_states_q = nn.functional.gelu(self.inhibition_q(query_states))
                threshold_inhi_q = torch.max(self.select_q(rehearsal_states_q)) * threshold_ratio
                selection_states_q = nn.functional.gelu(self.select_q(rehearsal_states_q) - threshold_inhi_q) + 1
                query_layer = self.transpose_for_scores(torch.mul(selection_states_q, self.query_proj(query_states)), self.num_attention_heads)
              else:
                query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
      else:
          query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)

      if self.gi_key_side:
          if self.num_attention_heads != self.gi_attention_heads_i:
              self.segment_num = self.num_attention_heads - self.gi_attention_heads_i

              key_states_head, key_states_tail = torch.split(hidden_states,
                                                              [self.segment_num, self.gi_attention_heads_i], dim=0)
              rehearsal_states_k = nn.functional.gelu(self.inhibition_k(key_states_tail))
              threshold_inhi_k = torch.max(self.select_k(rehearsal_states_k)) * threshold_ratio
              selection_states_k = nn.functional.gelu(self.select_k(rehearsal_states_k) - threshold_inhi_k) + 1
              key_layer_head = self.key_proj(key_states_head)
              key_layer_tail = torch.mul(selection_states_k, self.key_proj(key_states_tail))
              key_layer_temp = torch.cat((key_layer_head, key_layer_tail), 0)
              key_layer = self.transpose_for_scores(key_layer_temp, self.num_attention_heads)
          else:
              if self.layer_i > self.num_hidden_layers - self.last_layers_num:
                  rehearsal_states_k = nn.functional.gelu(self.inhibition_k(hidden_states))
                  threshold_inhi_k = torch.max(self.select_k(rehearsal_states_k)) * threshold_ratio
                  selection_states_k = nn.functional.gelu(self.select_k(rehearsal_states_k) - threshold_inhi_k) + 1
                  key_layer = self.transpose_for_scores(torch.mul(selection_states_k, self.key_proj(hidden_states)),
                                                        self.num_attention_heads)
              else:
                  key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
      else:
          key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)

      value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
```

### Run DeBERTa experiments from command line
For glue tasks, run task

``` bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=32000
#SBATCH --job-name=giDeBERTa30
#SBATCH --err=giDeBERTa30.err 
#SBATCH --out=giDeBERTa30.out 

/bin/hostname
srun -l /bin/hostname
srun -l /bin/pwd
ml PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
cd /home/user/gate_inhibition/python_script/DeBERTa-GLUE/
source EnvAMD/bin/activate
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name cola --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/CoLA/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name mrpc --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/MRPC/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name rte --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/RTE/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name wnli --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/WNLI/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name stsb --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/STSB/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name qqp --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/QQP/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name mnli --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/MNLI/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name qnli --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/QNLI/
python transformers/examples/pytorch/text-classification/run_glue.py --model_name_or_path microsoft/deberta-v3-large --task_name sst2 --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir --output_dir output_final/DeBERTa_30/SST2/
```


## Experiments
Our fine-tuning experiments are carried on half a DGX-2 node with 8x32 V100 GPU cards, the results may vary due to different GPU models, drivers, CUDA SDK versions, using FP16 or FP32, and random seeds. 
We report our numbers based on multple runs with different random seeds here. Here are the results from the Large model:

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI large|	`experiments/glue/mnli.sh large`|	91.1/91.1 +/-0.1|	2.5h|
|QQP large|	`experiments/glue/qqp.sh large`|	93.2 +/-0.1|		6h|
|QNLI large|	`experiments/glue/qnli.sh large`|	96.0 +/-0.3|		2h|
|MRPC large|	`experiments/glue/mrpc.sh large`|	93.1 +/-0.6|		0.5h|
|RTE large|	`experiments/glue/rte.sh large`|	92.1 +/-1.4|		0.5h|
|SST-2 large|	`experiments/glue/sst2.sh large`|	96.9 +/-0.3|		1h|
|STS-b large|	`experiments/glue/Stsb.sh large`|	93.2 +/-0.2|		0.5h|
|CoLA large|	`experiments/glue/cola.sh`|	76.5 +/-1.0|		0.5h|


### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and several GLUE benchmark tasks.

| Model                     | SQuAD 1.1 | SQuAD 2.0 | MNLI-m/mm   | SST-2 | QNLI | CoLA | RTE    | MRPC  | QQP   |STS-B |
|---------------------------|-----------|-----------|-------------|-------|------|------|--------|-------|-------|------|
|                           | F1/EM     | F1/EM     | Acc         | Acc   | Acc  | MCC  | Acc    |Acc/F1 |Acc/F1 |P/S   |
| BERT-Large                | 90.9/84.1 | 81.8/79.0 | 86.6/-      | 93.2  | 92.3 | 60.6 | 70.4   | 88.0/-       | 91.3/- |90.0/- |
| RoBERTa-Large             | 94.6/88.9 | 89.4/86.5 | 90.2/-      | 96.4  | 93.9 | 68.0 | 86.6   | 90.9/-       | 92.2/- |92.4/- |
| XLNet-Large               | 95.1/89.7 | 90.6/87.9 | 90.8/-      | 97.0  | 94.9 | 69.0 | 85.9   | 90.8/-       | 92.3/- |92.5/- |
| [DeBERTa-Large](https://huggingface.co/microsoft/deberta-large)| 95.5/90.1 | 90.7/88.0 | 91.3/91.1| 96.5|95.3| 69.5| 91.0| 92.6/94.6| 92.3/- |92.8/92.5 |
| [DeBERTa-XLarge](https://huggingface.co/microsoft/deberta-xlarge)| -/-  | -/-  | 91.5/91.2| 97.0 | - | -    | 93.1   | 92.1/94.3    | -    |92.9/92.7|
| [DeBERTa-V2-XLarge](https://huggingface.co/microsoft/deberta-v2-xlarge)|**95.8/90.8**| 91.4/88.9|91.7/91.6| **97.5**| 95.8|71.1|**93.9**|92.0/94.2|92.3/89.8|92.9/92.9|
|[DeBERTa-V3-Large](https://huggingface.co/microsoft/deberta-v3-large)|-/-|91.5/89.0|**91.8/91.9**|96.9|**96.0**|75.3| 92.7| 92.2/-|93.0/- |93.0/- |
| **giDeBERTaV3 <sup>1</sup>**|95.5/89.5|**91.6/89.0**|91.1/91.1|96.9|**96.0**|**76.5**|92.1|**93.1/-**|**93.2/-**|**93.2/-**|


--------
#### Notes.
 - <sup>1</sup> Following DeBERTaV3, we fine-tune GLUE tasks based on [DeBERTa-V3-Large](https://huggingface.co/microsoft/deberta-v3-large). When the inhibition percentile is 10\% or 30\%, the great improvements on CoLA (1.2\% over original DeBERTaV3 and 4.2\% over original DeBERTaV3 with extra 10 epochs fine-tuning), STS-B (0.2\% over original DeBERTaV3 and 0.6\% over original DeBERTaV3 with extra 10 epochs fine-tuning), QNLI (0.6\% over original DeBERTaV3 with extra 10 epochs fine-tuning), MRPC (0.9\% over original DeBERTaV3 and 2.4\% over original DeBERTaV3 with extra 10 epochs fine-tuning), as well as little improvement on QQP (0.2\%), SQuAD v2.0 (0.1\%) and SWAG (0.1\%) when the inhibition percentile is 30\% or 90\%.


 



