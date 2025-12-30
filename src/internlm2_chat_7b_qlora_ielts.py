# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = './internlm2-chat-7b'
use_varlen_attn = False

# Data
alpaca_en_path = './train_cn.json'
prompt_template = PROMPT_TEMPLATE.default
max_length = 1024
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_inputs = ['雅思作文题目是: Young people who commit crimes should be treated in the same way as adults who commit crimes. To what extent do you agree or disagree?, 作文内容是: Deciding to choose among the potential ways of punishing young people who commit crimes continues to be a controversial issue for the societies and the governments. It is argued by some that these people should be treated the same as adults. I personally disagree with this opinion due to the adverse effects of imprisonment on a teenager’s mental health. Many countries put the criminals of their society at jails considering it as a very effective way of punishment. It is understandable that this does exert a positive influence on decreasing crime in the society by putting the criminals in an unpleasant situation which they would mostly never wish to experience again. As a result, this could impede them from attempting crime in the future. For example, my friend who had been sent to jail for 2 months because of repeatedly committing traffic offends, has never committed the same crime since being released admitting that being in prison had been intolerable for her. However, I believe using the same way of punishment for youngsters would not be a wise idea. In fact, teenagers are at a very critical age in which the core of their personality is being shaped. There for, sending them to prison like adults as a way of punishment for their crimes, which are most often pity crimes, would actually expose them to other criminals who might have some serious personality disorders and this would adversely affect their personality as an adult in the future. To put in another way, such punishments are potential to become a threat to their mental health leading them to commit more serious crimes in a long run. For instance, according to the law of my country, young people are being punished the same as adults. A recent survey revealed that this policy has not been effective so far since 60% of these teenage criminals end up drug trafficking after being released from prison which had been sent to for a pity crime. In conclusion, although the ways that adults are being punished in many countries might be quite effective to decrease crime rate, I do not agree that it is a wise decision to use these ways for punishing young people as well., 请给出批改意见和评分', '雅思作文题目是: some people think that the best way to solve global environmental problems is to increase the cost of fuel. To what extent do you agree or disagree?, 作文内容是: It is widely believed that increasing the cost of fuel is the best way to solve the global environment. Personally, I completely disagree with this statement for a variety of reasons. First of all, the increase in the cost of fuel will lead to worldwide inflation. This means that all the daily stuff around you will have a rise in money. For example, food, accommodation, and health care,.. will be more pricey, more valued and more which is because the cash loses its own value because it is just printed automatically by the government to recover inflation. Furthermore, it slows down the evaluation of society because there are not enough space, or material for human to create or discover something new Secondly, transportation is strongly affected by the increase in the cost of fuel. This means that all kinds of transport through the sea or in the ground or in the sky can be cut off time seriously. Consequently, again, it leads to the growth of the cost of daily stuff which makes citizens in African or war countries situations become even worse than ever. They will have to deal with an impossible problem to solve and the result of that maybe is the disappearance of any country with a huge amount of victims. Moreover, it contributes to the economic depression all over the world which can create some things like World War three because there will be a crazy amount of people who may die because of the lack of food so it makes us have to fight for each other. In conclusion, increasing the cost of fuel is definitely not a great way to solve these prominent environmental problems. In my opinion, the value of fuel is not an aspect that we should care about, the thing that we should know is every transportation like cars or gas stoves,.. are used popularly so it creates an enormous amount of toxic gas., 请给出批改意见和评分', '雅思作文题目是: Topic: Some people believe that studying at university or college is the best route to a successful career, while others believe that it is better to get a job straight after school. Discuss both views., 作文内容是: After the high school education students encounter two main options: going to university or getting a job. Some people think that candidates who are new graduated should start college whereas a number of citizens assert finding work is better than college. It seems to me that two choices which are cited above have advantages and disadvantages. In this essay will be discussed these choices. On the one hand, universities supply many opportunities to the students for successful careers such as being a lawyer, teacher or politician. At the same time, these kinds of jobs are beneficial for society due to needs of high qualified occupations. For example, a student who chose the pursuing collage education, could receive bachelor degrees from school became a doctor and could serve for public. Therefore, university education is quite important for getting spectacular works. On the other hand, employees who are subsequence of highschool starting work, have been evaluated by numerous expert as essential figures of the labour market. For instance, many sectors which do not require diverse talent, need nonqualified workers. Moreover, employers who held the basic jobs can not offer work to high qualified people. Hence, getting a job after the secondary education valuable as going to university. Should be considered that many people can find job easily rather than qualified workers. To conclude, both options are very important for the benefit of community and each choice submit different career opportunity. Thus, options should be prefered by candidates for their tendencies and there are not an inequality between the two choices mentioned before., 请给出批改意见和评分']

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = {
    'type': AutoTokenizer.from_pretrained,
    'pretrained_model_name_or_path': pretrained_model_name_or_path,
    'trust_remote_code': True,
    'padding_side': 'right'
}

model = {
    'type': SupervisedFinetune,
    'use_varlen_attn': use_varlen_attn,
    'llm': {
        'type': AutoModelForCausalLM.from_pretrained,
        'pretrained_model_name_or_path': pretrained_model_name_or_path,
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
        'quantization_config': {
            'type': BitsAndBytesConfig,
            'load_in_4bit': True,
            'load_in_8bit': False,
            'llm_int8_threshold': 6.0,
            'llm_int8_has_fp16_weight': False,
            'bnb_4bit_compute_dtype': torch.float16,
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_quant_type': 'nf4'
        }
    },
    'lora': {
        'type': LoraConfig,
        'r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    }
}

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = {
    'type': process_hf_dataset,
    'dataset': {'type': load_dataset, 'path': 'json', 'data_files': {'train': alpaca_en_path}},
    'tokenizer': tokenizer,
    'max_length': max_length,
    'dataset_map_fn': openai_map_fn,
    'template_map_fn': {
        'type': template_map_fn_factory,
        'template': prompt_template
    },
    'remove_unused_columns': True,
    'shuffle_before_pack': True,
    'pack_to_max_length': pack_to_max_length,
    'use_varlen_attn': use_varlen_attn
}

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = {
    'batch_size': batch_size,
    'num_workers': dataloader_num_workers,
    'dataset': alpaca_en,
    'sampler': {'type': sampler, 'shuffle': True},
    'collate_fn': {'type': default_collate_fn, 'use_varlen_attn': use_varlen_attn}
}

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = {
    'type': AmpOptimWrapper,
    'optimizer': {
        'type': optim_type,
        'lr': lr,
        'betas': betas,
        'weight_decay': weight_decay
    },
    'clip_grad': {'max_norm': max_norm, 'error_if_nonfinite': False},
    'accumulative_counts': accumulative_counts,
    'loss_scale': 'dynamic',
    'dtype': 'float16'
}

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    {
        'type': LinearLR,
        'start_factor': 1e-5,
        'by_epoch': True,
        'begin': 0,
        'end': warmup_ratio * max_epochs,
        'convert_to_iter_based': True
    },
    {
        'type': CosineAnnealingLR,
        'eta_min': 0.0,
        'by_epoch': True,
        'begin': warmup_ratio * max_epochs,
        'end': max_epochs,
        'convert_to_iter_based': True
    }
]

# train, val, test setting
train_cfg = {'type': TrainLoop, 'max_epochs': max_epochs}

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    {'type': DatasetInfoHook, 'tokenizer': tokenizer},
    {
        'type': EvaluateChatHook,
        'tokenizer': tokenizer,
        'every_n_iters': evaluation_freq,
        'evaluation_inputs': evaluation_inputs,
        'system': SYSTEM,
        'prompt_template': prompt_template
    }
]

if use_varlen_attn:
    custom_hooks += [{'type': VarlenAttnArgsToMessageHubHook}]

# configure default hooks
default_hooks = {
    # record the time of every iteration.
    'timer': {'type': IterTimerHook},
    # print log every 10 iterations.
    'logger': {'type': LoggerHook, 'log_metric_by_epoch': False, 'interval': 10},
    # enable the parameter scheduler.
    'param_scheduler': {'type': ParamSchedulerHook},
    # save checkpoint per `save_steps`.
    'checkpoint': {
        'type': CheckpointHook,
        'by_epoch': False,
        'interval': save_steps,
        'max_keep_ckpts': save_total_limit
    },
    # set sampler seed in distributed evrionment.
    'sampler_seed': {'type': DistSamplerSeedHook},
}

# configure environment
env_cfg = {
    # whether to enable cudnn benchmark
    'cudnn_benchmark': False,
    # set multi process parameters
    'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0},
    # set distributed parameters
    'dist_cfg': {'backend': 'nccl'},
}

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = {'seed': None, 'deterministic': False}

# set log processor
log_processor = {'by_epoch': False}
