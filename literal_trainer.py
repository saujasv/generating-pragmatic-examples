import argparse
from dataclasses import dataclass, field
import torch
from typing import Optional
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoConfig,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    HfArgumentParser,
    GenerationConfig
)
from utils import DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    
    Adapted from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    random_init: bool = field(
        default=False,
        metadata={"help": "Whether to initialize the model with random weights or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    
    Adapted from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a tsv)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate validation loss."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

def train_listener(model_args, data_args, training_args):
    dataset_paths = {
        "train": [data_args.train_file],
        "validation": [data_args.validation_file]
    }

    dataset = load_dataset(
        "csv", 
        data_files=dataset_paths, 
        cache_dir=model_args.cache_dir, 
        sep="\t", 
        header=None, 
        column_names=["context", "bos_token", "target"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if model_args.random_init:
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )


    def preprocess_function(examples):
        model_inputs = tokenizer([' ' if x is None else x for x in examples["context"]], text_target=examples["target"])
        decoder_input_ids = [[bos, *inp[:-1]] for bos, inp in zip(tokenizer.convert_tokens_to_ids(examples["bos_token"]), model_inputs['labels'])]
        return {**model_inputs, 'decoder_input_ids': decoder_input_ids}
    
    tokenized_dataset = dataset.filter(lambda x: x['target'] is not None and x['context'] is not None).map(preprocess_function, batched=True, remove_columns=["target", "bos_token", "context"], batch_size=1000, num_proc=16)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    train_listener(model_args, data_args, training_args)
