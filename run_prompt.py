# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import shutil
import time
import json

import datasets
from datasets import load_dataset, load_metric
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from models import RobertaForPromptFinetuning

from utils import task_input_key, task_label_key, task_metric

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None, 
        required=True, 
        help="A dictionary containing the training, validation, test data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--shot_num", 
        type=int, 
        default=None, 
        required=True,
        help="The number of shots to use for training."    
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Select top k label token for each class."
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=None,
        help="The number of steps to use for evaluation."
    )
    parser.add_argument(
        "--logging_loss_steps", 
        type=int, 
        default=10,
        help="The number of steps to use for logging the loss."
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default=None,
        help="The template to use for the output file."
    )
    parser.add_argument(
        "--dedup", 
        action="store_true",
        default=False,
        help="Whether to dedup label tokens."
    )
    parser.add_argument(
        "--random_k_token", 
        action='store_true', 
        default=False,
        help="Whether to random select k label tokens."
    )
    parser.add_argument(
        "label_token_mode",
        type=str,
        choices=["AMuLaP", "AutoL", "PETAL"],
        default="AMuLaP",
        help="How to get the label token."
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default=None,
        help="The path to the label token mapping file."
    )
    parser.add_argument(
        "--max_seq_len", 
        type=int, 
        default=128,
        help="The maximum sequence length."
    )
    parser.add_argument(
        "--first_sent_limit", 
        type=int, 
        default=None,
        help="The maximum first sentence length."
    )
    parser.add_argument(
        "--other_sent_limit", 
        type=int, 
        default=None,
        help="The maximum other sentence length."
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        default=False,
        help="Whether to finetune the model."
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.logging_dir = os.path.join(args.output_dir, "logging", args.task_name, str(args.shot_num) + "-" + str(args.seed))
        os.makedirs(args.logging_dir, exist_ok=True)
        if not args.no_finetune:
            dir_name = "trainstep{}_warmupstep{}_lr{}_pbs{}".format(args.max_train_steps, args.num_warmup_steps, args.learning_rate, args.per_device_train_batch_size)
            dir_name += "_topk{}".format(args.top_k)
            dir_name += "_" + args.label_token_mode
            if args.label_token_mode == "AMuLaP":
                dir_name += "_random" if args.random_k_token else ""
                dir_name += "_dedup" if args.dedup else ""
            args.output_dir = os.path.join(args.output_dir, args.task_name, str(args.shot_num) + "-" + str(args.seed), dir_name)
            os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def load_data(task_name, data_dir):
    if task_name in ["sst2", "cola", "mrpc", "qnli", "qqp", "rte"]:
        data_files = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev.tsv"),
            "test": os.path.join(data_dir, "test.tsv"),
        }
    elif task_name in ["mnli"]:
        data_files = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev_matched.tsv"),
            "test_m": os.path.join(data_dir, "test_matched.tsv"),
            "test_mm": os.path.join(data_dir, "test_mismatched.tsv"),
        }
        
    if task_name in ["sst2", "mnli", "mrpc", "qnli", "qqp", "rte"]:
        dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)
    elif task_name in ["cola"]:
        dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=["id", "label", "_", "sentence"])

    return dataset

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    # logging_dir = os.path.join(args.output_dir, "logging", args.task_name, str(args.shot_num) + "-" + str(args.seed))
    # os.makedirs(logging_dir, exist_ok=True)
    filename = None
    if args.no_finetune:
        filename = "no_finetune"
    else:
        filename = "trainstep{}_warmupstep{}_lr{}_pbs{}".format(args.max_train_steps, args.num_warmup_steps, args.learning_rate, args.per_device_train_batch_size)
    filename += "_topk{}".format(args.top_k)
    filename += "_" + args.label_token_mode
    if args.label_token_mode == "AMuLaP":
        filename += "_random" if args.random_k_token else ""
        filename += "_dedup" if args.dedup else ""
    filename += ".log"
    logging_filename = os.path.join(args.logging_dir, filename)
    logging.basicConfig(
        filename=logging_filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    raw_datasets = load_data(args.task_name, args.data_dir)

    label2id = None
    labels = raw_datasets["train"][task_label_key[args.task_name]]
    labels = list(set(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2id), finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = RobertaForPromptFinetuning.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    def tokenize_multipart_input(
        input_text_list,
        max_length,
        prompt=False, 
        template=None,
        label_word_list=None,
        first_sent_limit=None,
        other_sent_limit=None,
        gpt3=False,
        truncate_head=False,
        support_labels=None,
    ):
        def enc(text):
            return tokenizer.encode(text, add_special_tokens=False)

        input_ids = []
        attention_mask = []
        token_type_ids = [] # Only for BERT
        mask_pos = None # Position of the mask token

        if prompt:
            """
            Concatenate all sentences and prompts based on the provided template.
            Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
            *xx* represent variables:
                *cls*: cls_token
                *mask*: mask_token
                *sep*: sep_token
                *sep+*: sep_token, also means +1 for segment id
                *sent_i*: sentence i (input_text_list[i])
                *sent-_i*: same as above, but delete the last token
                *sentl_i*: same as above, but use lower case for the first word
                *sentl-_i*: same as above, but use lower case for the first word and delete the last token
                *+sent_i*: same as above, but add a space before the sentence
                *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
                *label_i*: label_word_list[i]
                *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

            Use "_" to replace space.
            PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
            """
            assert template is not None

            special_token_mapping = {
                'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
            }
            template_list = template.split('*') # Get variable list in the template
            segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

            for part_id, part in enumerate(template_list):
                new_tokens = []
                segment_plus_1_flag = False
                if part in special_token_mapping:
                    if part == 'cls' and 'T5' in type(tokenizer).__name__:
                        # T5 does not have cls token
                        continue
                    new_tokens.append(special_token_mapping[part])
                    if part == 'sep+':
                        segment_plus_1_flag = True
                elif part[:6] == 'label_':
                    # Note that label_word_list already has extra space, so do not add more space ahead of it.
                    label_id = int(part.split('_')[1])
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                elif part[:7] == 'labelx_':
                    instance_id = int(part.split('_')[1])
                    label_id = support_labels[instance_id]
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                elif part[:5] == 'sent_':
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(input_text_list[sent_id]) 
                elif part[:6] == '+sent_':
                    # Add space
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(' ' + input_text_list[sent_id])
                elif part[:6] == 'sent-_':
                    # Delete the last token
                    sent_id = int(part.split('_')[1])
                    new_tokens += enc(input_text_list[sent_id][:-1])
                elif part[:6] == 'sentl_':
                    # Lower case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == '+sentl_':
                    # Lower case the first token and add space 
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(' ' + text)
                elif part[:7] == 'sentl-_':
                    # Lower case the first token and discard the last token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text[:-1])
                elif part[:6] == 'sentu_':
                    # Upper case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == '+sentu_':
                    # Upper case the first token and add space
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(' ' + text)
                else:
                    # Just natural language prompt
                    part = part.replace('_', ' ') 
                    # handle special case when T5 tokenizer might add an extra space
                    if len(part) == 1:
                        new_tokens.append(tokenizer._convert_token_to_id(part))
                    else:
                        new_tokens += enc(part)

                if part[:4] == 'sent' or part[1:5] == 'sent':
                    # If this part is the sentence, limit the sentence length
                    sent_id = int(part.split('_')[1])
                    if sent_id == 0:
                        if first_sent_limit is not None:
                            new_tokens = new_tokens[:first_sent_limit]
                    else:
                        if other_sent_limit is not None:
                            new_tokens = new_tokens[:other_sent_limit]

                input_ids += new_tokens
                attention_mask += [1 for i in range(len(new_tokens))]
                token_type_ids += [segment_id for i in range(len(new_tokens))]

                if segment_plus_1_flag:
                    segment_id += 1
        else:
            input_ids = [tokenizer.cls_token_id]
            attention_mask = [1]
            token_type_ids = [0]

            for sent_id, input_text in enumerate(input_text_list):
                if input_text is None:
                    # Do not have text_b
                    continue
                if pd.isna(input_text) or input_text is None:
                    # Empty input
                    input_text = ''
                input_tokens = enc(input_text) + [tokenizer.sep_token_id]
                input_ids += input_tokens
                attention_mask += [1 for i in range(len(input_tokens))]
                token_type_ids += [sent_id for i in range(len(input_tokens))]

            if 'T5' in type(tokenizer).__name__: # T5 does not have CLS token
                input_ids = input_ids[1:]
                attention_mask = attention_mask[1:]
                token_type_ids = token_type_ids[1:]

        # Padding
        if first_sent_limit is not None and len(input_ids) > max_length:
            # If using sentence limit, the total length still exceeds the maximum limit, report a warning
            logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

        while len(input_ids) < max_length:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(0)

        # Truncate
        if len(input_ids) > max_length:
            if truncate_head:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                token_type_ids = token_type_ids[-max_length:]
            else:
                # Default is to truncate the tail
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]

        # Find mask token
        if prompt:
            mask_pos = [input_ids.index(tokenizer.mask_token_id)]
            # Make sure that the masked position is inside the max_length
            assert mask_pos[0] < max_length

        result = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'BERT' in type(tokenizer).__name__:
            # Only provide token type ids for BERT
            result['token_type_ids'] = token_type_ids

        if prompt:
            result['mask_pos'] = mask_pos

        return result

    def preprocess_function(examples):
        # Tokenize the texts
        result = {}
        result["input_ids"] = []
        result["attention_mask"] = []
        result["mask_pos"] = []

        if len(task_input_key[args.task_name]) == 1:
            sentences = examples[task_input_key[args.task_name][0]]
            input_text_lists = [[sent] for sent in sentences]
        else:
            sentences1 = examples[task_input_key[args.task_name][0]]
            sentences2 = examples[task_input_key[args.task_name][1]]
            input_text_lists = [[sent1, sent2] for sent1, sent2 in zip(sentences1, sentences2)]
        
        for input_text_list in input_text_lists:
            res = tokenize_multipart_input(
                input_text_list=input_text_list,
                max_length=args.max_seq_len,
                prompt=True,
                template=args.template,
                first_sent_limit=args.first_sent_limit,
                other_sent_limit=args.other_sent_limit,
                )
            
            result["input_ids"].append(res["input_ids"])
            result["attention_mask"].append(res["attention_mask"])
            result["mask_pos"].append(res["mask_pos"])

        if task_label_key[args.task_name] in examples:
            result["labels"] = [label2id[label] for label in examples[task_label_key[args.task_name]]]
        
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["dev"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # get top-k token index
    k_map = {}
    if args.label_token_mode == "AMuLaP":
        model.eval()
        all_train_logits = {}
        with torch.no_grad():
            for step, batch in enumerate(train_dataloader):
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
                for i in range(len(batch["labels"])):
                    if batch["labels"][i].item() not in all_train_logits:
                        all_train_logits[batch["labels"][i].item()] = outputs[0][i].cpu()
                    else:
                        all_train_logits[batch["labels"][i].item()] += outputs[0][i].cpu()

        map_index = {}
        for key in label2id:
            label = label2id[key]
            all_train_logits[label] = all_train_logits[label] / args.shot_num
            sorted_logits, sort_index = torch.sort(all_train_logits[label], descending=True)
            map_index[label] = sort_index.tolist()
        
        label_token_set = {}
        for key in label2id:
            label_token_set[label2id[key]] = []

        for i in range(tokenizer.vocab_size):
            logits = [all_train_logits[label2id[key]][i] for key in label2id]
            label_token_set[logits.index(max(logits))].append({
                "idx": i,
                "prob": max(logits),
            })

        def myFunc(e):
            return e['prob']
    
        for key in label2id:
            label = label2id[key]
            label_token_set[label].sort(reverse=True, key=myFunc)

        if args.dedup:
            for key in label2id:
                label = label2id[key]
                k_map[label] = []
                for i in range(args.top_k):
                    k_map[label].append(label_token_set[label][i]["idx"])
        elif args.random_k_token:
            num_list = random.sample(range(tokenizer.vocab_size), args.top_k * len(label2id))
            for i, key in enumerate(label2id):
                label = label2id[key]
                k_map[label] = num_list[i * args.top_k : (i+1) * args.top_k]
        else:
            for key in label2id:
                label = label2id[key]
                k_map[label] = map_index[label][:args.top_k]
    elif args.label_token_mode == "AutoL":
        label_to_word = {}
        for key in label2id:
            label = label2id[key]
            label_to_word[label] = []
        # seed_mapping_path = os.path.join(args.mapping_path, "{}-{}.sort.txt".format(args.shot_num, args.seed))
        with open(args.mapping_path) as f:
            for line in f:
                line = line.strip()
                line = eval(line)
                for key in line:
                    word = line[key]
                    if word[0] not in ['<', '[', '.', ',']:
                        assert len(tokenizer.tokenize(' ' + word)) == 1
                        word = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + word)[0])
                    else:
                        word = tokenizer._convert_token_to_id(word)
                    if len(key) == 1:
                        label_to_word[label2id[int(key)]].append(word)
                    else:
                        label_to_word[label2id[key]].append(word)
        for key in label2id:
            label = label2id[key]
            k_map[label] = label_to_word[label][:args.top_k]
    elif args.label_token_mode == "PETAL":
        label_to_word = {}
        for key in label2id:
            label = label2id[key]
            label_to_word[label] = []
        # seed_mapping_path = os.path.join(args.mapping_path, "{}-{}.json".format(args.shot_num, args.seed))
        with open(args.mapping_path) as f:
            for line in f:
                line = line.strip()
                line = json.loads(line)
                for key in line:
                    for word in line[key]:
                        if word[0] not in ['<', '[', '.', ',']:
                            if len(tokenizer.tokenize(' ' + word)) == 1:
                                word = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + word)[0])
                            else:
                                word = tokenizer._convert_token_to_id(word)
                        else:
                            word = tokenizer._convert_token_to_id(word)
                        
                        if len(key) == 1:
                            label_to_word[label2id[int(key)]].append(word)
                        else:
                            label_to_word[label2id[key]].append(word)
        for key in label2id:
            label = label2id[key]
            k_map[label] = label_to_word[label][:args.top_k]

    model.label_token_list = {}
    for key in label2id:
        label = label2id[key]
        model.label_token_list[label] = torch.tensor(k_map[label]).long().cuda()

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    metric = load_metric("glue", args.task_name)

    if args.no_finetune:
        for split in ["dev", "test"] if args.task_name != "mnli" else ["dev", "test_m", "test_mm"]:
            eval_dataset = processed_datasets[split]
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
                predictions = outputs[0].argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"{split}: {eval_metric}")
        return

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_metric = -1
    best_metric_step = None

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"], batch["labels"])
            loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % args.logging_loss_steps == 0 and completed_steps != 0:
                logger.info(f"step {completed_steps}: loss: {loss}")

            if completed_steps >= args.max_train_steps:
                break

            if args.eval_steps is not None:
                if completed_steps % args.eval_steps == 0 and completed_steps != 0:
                    model.eval()
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        outputs = model(eval_batch["input_ids"], eval_batch["attention_mask"], eval_batch["mask_pos"])
                        predictions = outputs[0].argmax(dim=-1)
                        metric.add_batch(
                            predictions=accelerator.gather(predictions),
                            references=accelerator.gather(eval_batch["labels"]),
                        )

                    eval_metric = metric.compute()
                    logger.info(f"step {completed_steps}: {eval_metric}")
                    if eval_metric[task_metric[args.task_name]] > best_metric:
                        best_metric = eval_metric[task_metric[args.task_name]]
                        best_metric_step = completed_steps
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                    model.train()

    model.eval()
    for eval_step, eval_batch in enumerate(eval_dataloader):
        outputs = model(eval_batch["input_ids"], eval_batch["attention_mask"], eval_batch["mask_pos"])
        predictions = outputs[0].argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(eval_batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"step {completed_steps}: {eval_metric}")
    if eval_metric[task_metric[args.task_name]] > best_metric:
        best_metric = eval_metric[task_metric[args.task_name]]
        best_metric_step = completed_steps
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    logger.info(f"early stop at step {best_metric_step}, metric: {best_metric}")
    
    model = RobertaForPromptFinetuning.from_pretrained(
        args.output_dir, 
        from_tf=bool(".ckpt" in args.output_dir), 
        config=config,
        )

    model.label_token_list = {}
    for key in label2id:
        label = label2id[key]
        model.label_token_list[label] = torch.tensor(k_map[label]).long().cuda()
    model = accelerator.prepare(model)

    for split in ["test"] if args.task_name != "mnli" else ["test_m", "test_mm"]:
        eval_dataset = processed_datasets[split]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["mask_pos"])
            predictions = outputs[0].argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"{split}: {eval_metric}")

    # Delete the saved outputs to save space
    shutil.rmtree(args.output_dir)

if __name__ == "__main__":
    main()