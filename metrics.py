import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, GenerationConfig, DataCollatorForSeq2Seq
from datasets import load_dataset
from copy import deepcopy
from utils import get_preprocess_function, byt5_decode_batch, get_utterance_processing_functions

def compute_label_ratio_candidates(utterances_file):
    """
        Take in an utterances_file, and for each program compute how many 
        of the proposed utterances have positive label and how many have 
        negative label. Return a list where each element is the proportion of
        positive candidates for that target and context.
    """

    utterances_to_string, string_to_utterances = get_utterance_processing_functions("suffix", True)

    ratios = []
    with open(utterances_file, 'r') as f:
        for line in f:
            program, context, candidates_str = line.strip('\n').split('\t')
            candidates = candidates_str.split(' ')
            n_pos, n_neg = 0, 0
            for u in candidates:
                string, label = string_to_utterances(u)[0]
                if label == '+':
                    n_pos += 1
                elif label == '-':
                    n_neg += 1
                else:
                    raise ValueError("Invalid example string")
        
            ratios.append(n_pos / (n_pos + n_neg))
    
    return ratios

def compute_label_ratio_specs(contexts_file, max_len=10):
    """
        Take in contexts_file and for each program-spec pair compute the 
        ratio of positive and negative labels at each position. Return a 
        list where the ith element is the proportion of entries in
        contexts_file where the ith utterance is a positive example.
    """
    
    utterances_to_string, string_to_utterances = get_utterance_processing_functions("prefix", False)
    ratios = [{'pos': 0, 'neg': 0} for i in range(max_len)]
    with open(contexts_file, 'r') as f:
        for line in f:
            program, context = line.strip('\n').split('\t')
            spec = context.split(' ')
            for u, counts in zip(spec, ratios):
                string, label = string_to_utterances(u)[0]
                if label == '+':
                    counts['pos'] += 1 
                elif label == '-':
                    counts['neg'] += 1 
                else:
                    raise ValueError("Invalid example string")
    
    return [r['pos'] / (r['pos'] + r['neg']) if r['pos'] + r['neg'] > 0 else None for r in ratios]

def compute_model_ppl(
        tokenizer, model, 
        contexts_file, batch_size=4
        ):
    
    dataset = load_dataset(
        "csv", 
        data_files={"eval": contexts_file},
        sep="\t", 
        header=None, 
        column_names=["target", "context"]
    )

    tokenized_dataset = dataset.map(
        get_preprocess_function(tokenizer),
        batched=True, remove_columns=["target", "context"]
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        padding=True,
        label_pad_token_id=model.config.pad_token_id
    )

    dataloader = DataLoader(
        tokenized_dataset["eval"], batch_size=batch_size, 
        shuffle=False, collate_fn=collator
        )
    losses = list()
    for batch in dataloader:
        outputs = model(**batch, return_dict=True)
        loss = F.cross_entropy(
            outputs.logits, batch["labels"], 
            reduction="none", ignore_index=model.config.pad_token_id
            )
        losses.extend(loss.sum(dim=1).tolist())

    return losses

# def self_play_score(
#         listener_model, listener_gen_config, 
#         speaker_model, speaker_gen_config, 
#         program_targets, max_turns=8, 
#         speaker_device="cuda:0", listener_device="cuda:0"
#         ):
#     # TODO: Play communication game between listener model and speaker model 
#     # over all programs in program_targets for up to max_turns turns and 
#     # return success metric

#     for program in program_targets:
#         context = []
#         for turn_idx in range(max_turns):
#             spk_tokens = tokenizer(f"{program} {' '.join(context)}", return_tensors="pt").to(speaker_device)
#             spk_outputs = speaker_model.generate(**spk_tokens, generation_config=speaker_gen_config)
#             spk_utterance = byt5_decode_batch(spk_outputs, skip_position_token=True, skip_special_tokens=True)

if __name__ == "__main__":
    for i in range(1, 5):
        print(np.mean(compute_label_ratio_candidates(f"/compute/tir-0-19/svadugur/working-50-50-1024-32-5-1/round-0-{i}-train-utterances.tsv")))
    # print(compute_label_ratio_specs("data/listener-eval-set-users.tsv"))
    # tokenizer = AutoTokenizer.from_pretrained("/scratch/svadugur/literal-listener-3-reweighted-5M-156k")
    # with open("configs/inference_params/beam_search_bs=50_lp=-1.json") as f:
    #     gen_config = json.load(f)
    #     print(gen_config)
    # # source_model=None
    # target_model=None

    # torch.cuda.empty_cache()
    # source_model = T5ForConditionalGeneration.from_pretrained("/scratch/svadugur/literal-listener-3-reweighted-5M-156k").to("cuda:0")
    # # target_model = T5ForConditionalGeneration.from_pretrained("/scratch/svadugur/listener-32768/checkpoint-25125").to("cuda:0")

    # print("starting")
    # compute_model_contrast(tokenizer, GenerationConfig(**gen_config), source_model, target_model, "working-50-50-32768-1024-5-1/round-0-5-validation-contexts.tsv")
