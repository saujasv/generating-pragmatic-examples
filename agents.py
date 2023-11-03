from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, get_scheduler, LogitsProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import torch
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from moe_decoding import MoELogitsProcessor, AdditiveMixture
from gumbel_noise import GumbelLogitsProcessor
from utils import get_preprocess_function, get_utterance_processing_functions, byt5_decode_batch, consistent, DataCollatorForSeq2Seq
from example_sampler import PROGRAM_SPECIAL_TOKEN, UTTERANCES_SPECIAL_TOKEN, GT_PROGRAM_SPECIAL_TOKEN
from args import TrainingArguments
import logging
from pathlib import Path
import random
import os
from greenery import parse
from greenery.parse import NoMatch
import numpy as np

class Agent:
    def __init__(self, 
                model_path: str, 
                trainable: bool, 
                save_path: str, 
                gen_config: dict, 
                inference_batch_size: int = 1,
                device: str = "cuda", 
                step: int = 0, 
                name: str = "",
                training_args: Optional[dict] = None,
                resume=None
                ):
        
        logging.info(f"Loading model {name} from {model_path}")
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.init_model_path = model_path
        self.most_recent_checkpoint = model_path
        self.step = step
        self.name = name
        self.gen_config = GenerationConfig(**gen_config)
        self.inference_batch_size = inference_batch_size
        self.trainable = trainable

        if self.trainable:
            self.save_path = save_path
            if resume is not None:
                self.most_recent_checkpoint = str(Path(self.save_path) / f"round-{resume - 1}")
            self.training_args = TrainingArguments(**training_args)
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.training_args.learning_rate
                )
            self.lr_scheduler = get_scheduler(
                self.training_args.lr_scheduler_type, self.optimizer, 
                self.training_args.num_warmup_steps
                )

    def evaluate_loss(self, dataset_paths):
        dataset = load_dataset(
            "csv", 
            data_files=dataset_paths,
            sep="\t", 
            header=None, 
            column_names=["context", "decoder_start_token", "target"]
        )

        def preprocess_function(examples):
            model_inputs = self.tokenizer(
                [' ' if x is None else x for x in examples["context"]], 
                text_target=examples["target"]
                )
            decoder_input_ids = [
                [bos, *inp[:-1]] 
                for bos, inp in zip(
                    self.tokenizer.convert_tokens_to_ids(examples["decoder_start_token"]), 
                    model_inputs['labels']
                    )]
            return {**model_inputs, 'decoder_input_ids': decoder_input_ids}

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True, remove_columns=["context", "decoder_start_token", "target"]
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model, 
            padding=True, return_tensors="pt"
            )

        split_losses = dict()
        for split in tokenized_dataset:
            dataloader = DataLoader(
                tokenized_dataset[split], 
                shuffle=True, 
                batch_size=self.inference_batch_size, 
                collate_fn=data_collator
                )
            loss_fct = CrossEntropyLoss(reduction="sum", ignore_index=-100)
            loss_aggregate = 0
            n_elements_aggregate = 0

            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                x = torch.ones(batch["labels"].shape)
                mask = (batch["labels"] == -100).to("cpu")
                n_elements = x.masked_fill(mask, 0).sum()

                loss_aggregate += loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)), 
                    batch["labels"].view(-1)
                    )
                n_elements_aggregate += n_elements
            
            split_losses[split] = loss_aggregate.item() / n_elements_aggregate.item()
        
        return split_losses
    
    def reload_model(self, from_init=False):
        if from_init:
            print(f"Loading model {self.name} from {self.init_model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.init_model_path).to(self.device)
            if self.trainable:
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.training_args.learning_rate
                    )
                self.lr_scheduler = get_scheduler(
                    self.training_args.lr_scheduler_type, self.optimizer, 
                    self.training_args.num_warmup_steps
                    )
        else:
            print(f"Loading model {self.name} from {self.most_recent_checkpoint}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.most_recent_checkpoint).to(self.device)
    
    def update_model(self, dataset_paths, save_path, validation_split="user_validation", save_every=None):
        dataset = load_dataset(
            "csv", 
            data_files=dataset_paths,
            sep="\t", 
            header=None, 
            column_names=["context", "decoder_start_token", "target"]
        )

        def preprocess_function(examples):
            model_inputs = self.tokenizer(
                [' ' if x is None else x for x in examples["context"]], 
                text_target=examples["target"]
                )
            decoder_input_ids = [
                [bos, *inp[:-1]] 
                for bos, inp in zip(
                    self.tokenizer.convert_tokens_to_ids(examples["decoder_start_token"]), 
                    model_inputs['labels']
                    )]
            return {**model_inputs, 'decoder_input_ids': decoder_input_ids}

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True, remove_columns=["target", "decoder_start_token", "context"]
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
            )

        train_dataloader = DataLoader(
            tokenized_dataset["train"], shuffle=True, 
            batch_size=self.training_args.train_batch_size, 
            collate_fn=data_collator
        )

        training_losses = []
        self.optimizer.zero_grad()

        # NOTE: no multi-epoch training for now
        for i, batch in enumerate(tqdm(train_dataloader)):
            if save_every is not None and i % save_every == 0:
                self.model.save_pretrained(os.path.join(save_path, f"step_{self.step}.pt"))
                self.tokenizer.save_pretrained(os.path.join(save_path, f"step_{self.step}.pt"))

            self.model.train()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            training_losses.append(loss.item())
            wandb.log({f"inner_loop/{self.name}_train_loss": loss.item(), f"{self.name}_step": self.step})
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.most_recent_checkpoint = save_path

@dataclass
class ListenerOutput:
    programs: List[List[str]]
    idx: Optional[List[List[int]]] = None
    decoded: Optional[List[List[str]]] = None
    decoded_scores: Optional[List[List[float]]] = None
    pruned: Optional[List[List[str]]] = None

class Listener(Agent):
    def __init__(self, 
        model_path,
        trainable, 
        gen_config, 
        save_path=None, 
        inference_batch_size=4,
        label_pos="suffix",
        idx: bool=True,
        device: str ="cuda:0", 
        step: int = 0, 
        name: str = "",
        training_args: Optional[dict] = None,
        program_special_token=PROGRAM_SPECIAL_TOKEN,
        utterances_special_token=UTTERANCES_SPECIAL_TOKEN,
        resume=None
    ):
        super().__init__(
            model_path, 
            trainable, 
            save_path, 
            gen_config, 
            inference_batch_size,
            device, 
            step, 
            name,
            training_args,
            resume
        )
        self.label_pos = label_pos
        self.idx = idx
        self.program_special_token = program_special_token
        self.utterances_special_token = utterances_special_token
        self.utterances_to_string, self.string_to_utterances = (
            get_utterance_processing_functions(
                label_pos, idx, separator=utterances_special_token
                )
            )
    
    def synthesize(self, context, return_scores=False, enforce_consistency=True):
        # If context is a list of utterances, convert to string
        if isinstance(context[0], list):
            context_str = list(map(self.utterances_to_string, context))
        else:
            context_str = context

        context_tokens = self.tokenizer(
            [f"{self.utterances_special_token}{c}" if not c.startswith(self.utterances_special_token) else c 
            for c in context_str], 
            return_tensors="pt",
            padding=True
            ).to(self.device)
        
        decoder_inputs = self.tokenizer(
            [self.program_special_token for _ in context], return_tensors="pt",
            add_special_tokens=False
            ).to(self.device)

        outputs = self.model.generate(**context_tokens, 
                                      decoder_input_ids=decoder_inputs.input_ids,
                                      generation_config=self.gen_config, 
                                      return_dict_in_generate=True, 
                                      output_scores=True
                                      )

        decoded_batch = byt5_decode_batch(outputs.sequences.reshape((len(context), -1, outputs.sequences.shape[-1])).tolist(), skip_position_token=True, skip_special_tokens=True)

        consistent_programs = []
        idxs = []
        for decoded, ctx in zip(decoded_batch, context):
            cp = []
            idx = []
            for i, p in enumerate(decoded):
                if enforce_consistency:
                    if consistent(p, ctx):
                        cp.append(p)
                        idx.append(i)
                else:
                    cp.append(p)
                    idx.append(i)
            
            consistent_programs.append(cp)
            idxs.append(idx)
        
        logprobs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
        gen_probs = torch.gather(logprobs, 2, outputs.sequences[:, 1:, None]).squeeze(-1)
        gen_probs.masked_fill_(gen_probs.isinf(), 0)
        scores = gen_probs.sum(-1)
        n_decoded = scores.shape[0]
        n_seq = n_decoded // len(context)
        scores = scores.reshape((len(context), n_seq))
        scores_list = scores.tolist()

        if return_scores:
            return ListenerOutput(
                consistent_programs,
                idxs, 
                decoded_batch, 
                scores_list
                )
        else:
            return ListenerOutput(consistent_programs)
    
    def prune_programs(self, programs):
        parsed = []
        for p in programs:
            try:
                parsed.append(parse(p))
            except NoMatch:
                pass

        import ipdb; ipdb.set_trace()
        pruned = list()
        for p in parsed:
            for q in pruned:
                if p.equivalent(q):
                    continue
                else:
                    pruned.append(p.reduce())
        
        return [str(p) for p in pruned]
    
    def score_program(self, contexts, programs):
        if isinstance(contexts[0], list):
            context_str = list(map(self.utterances_to_string, contexts))
        else:
            context_str = contexts

        context_tokens = self.tokenizer(
            [f"{self.utterances_special_token}{c}" if not c.startswith(self.utterances_special_token) else c 
            for c in context_str], 
            return_tensors="pt",
            padding=True
            ).to(self.device)

        program_tokens = self.tokenizer([f"{self.program_special_token}{p}" for p in programs], return_tensors="pt").to(self.device)
        outputs = self.model(input_ids=context_tokens.input_ids, decoder_input_ids=program_tokens.input_ids, return_dict=True)
        
        logprobs = torch.gather(F.log_softmax(outputs.logits, dim=-1), 2, program_tokens.input_ids[:, 1:, None]).squeeze(-1)
        
        logprobs.masked_fill_(program_tokens.input_ids[:, 1:] == 0, 0)

        scores = logprobs.sum(-1)
        
        return scores.tolist()
    
    def write_formatted_training_data(self, programs, contexts, output_file, shuffle=True, len_filter=None):
        utterances_to_string_no_idx, _ = get_utterance_processing_functions(self.label_pos, False)
        write_buffer = list()
        
        for prog, ctx in zip(programs, contexts):
            for i in range(1, len(ctx) + 1):
                if len_filter is not None and i != len_filter:
                    continue

                write_buffer.append((
                    f"{self.utterances_special_token}{self.utterances_to_string(ctx[:i])}", self.program_special_token, prog
                    ))
        
        if shuffle:
            random.shuffle(write_buffer)

        with open(output_file, 'w') as f:
            for inp, decoder_start_token, out in write_buffer:
                f.write(f"{inp}\t{decoder_start_token}\t{out}\n")
        
        return len(write_buffer)

@dataclass
class SpeakerOutput:
    utterances: List[List[tuple[str, str]]]
    idx: Optional[List[List[int]]] = None
    decoded: Optional[List[List[str]]] = None
    decoded_scores: Optional[List[float]] = None

class Speaker(Agent):
    def __init__(self, 
        model_path,
        trainable, 
        gen_config, 
        save_path=None, 
        inference_batch_size=4,
        label_pos="prefix",
        idx=True,
        device="cuda:0", 
        step=0, 
        name="",
        training_args=None,
        use_program=True,
        use_context=True,
        program_special_token=PROGRAM_SPECIAL_TOKEN,
        utterances_special_token=UTTERANCES_SPECIAL_TOKEN,
        type="std",
        resume=None
    ):
        super().__init__(model_path, trainable, save_path, gen_config, inference_batch_size, device, step, name, training_args, resume)
        self.label_pos = label_pos
        self.idx = idx
        self.utterances_to_string, self.string_to_utterances = get_utterance_processing_functions(label_pos, idx)
        self.use_program = use_program
        self.use_context = use_context
        self.program_special_token = program_special_token
        self.utterances_special_token = utterances_special_token
    
    def generate(self, targets, contexts, return_scores=False):
        assert len(targets) == len(contexts), "Number of targets and contexts must be the same"
        inputs = ['' for i in range(len(targets))]
        if self.use_program:
            inputs = [inp + f"{self.program_special_token}{target}" for inp, target in zip(inputs, targets)]
        if self.use_context:
            inputs = [inp + f"{self.utterances_special_token}{self.utterances_to_string(context)}" for inp, context in zip(inputs, contexts)]

        context_tokens = self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [f"<extra_id_{len(ctx)}>" if self.idx else self.utterances_special_token for ctx in contexts]
        decoder_start_token_ids = self.tokenizer(decoder_start_tokens, return_tensors="pt", add_special_tokens=False).to(self.device)
        outputs = self.model.generate(
            inputs=context_tokens.input_ids, 
            decoder_input_ids=decoder_start_token_ids.input_ids,
            generation_config=self.gen_config,
            return_dict_in_generate=True,
            output_scores=True
            )

        decoded_batch = byt5_decode_batch(
            outputs.sequences.reshape((len(contexts), -1, outputs.sequences.shape[-1])).tolist(), 
            skip_special_tokens=True, 
            skip_position_token=True
            )

        consistent_utterances = list()
        idxs = list()
        for target, ctx, decoded in zip(targets, contexts, decoded_batch):
            cu = list()
            idx = list()
            for i, utterance in enumerate(decoded):
                u = self.string_to_utterances(utterance)
                if u is None or len(u) == 0:
                    continue

                s, label = u[0]
                if len(s) > 25:
                    continue

                if not label in ['+', '-']:
                    continue

                if u[0] in ctx:
                    continue

                if ' ' in s or '\t' in s or '\n' in s:
                    continue

                if consistent(target, u):
                    cu.append(u[0])
                    idx.append(i)
            
            consistent_utterances.append(cu)
            idxs.append(idx)
        
        logprobs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
        gen_probs = torch.gather(logprobs, 2, outputs.sequences[:, 1:, None]).squeeze(-1)
        gen_probs.masked_fill_(gen_probs.isinf(), 0)
        scores = gen_probs.sum(-1)
        n_decoded = scores.shape[0]
        n_seq = n_decoded // len(contexts)
        scores = scores.reshape((len(contexts), n_seq))
        scores_list = scores.tolist()

        if return_scores:
            return SpeakerOutput(
                consistent_utterances,
                idxs, 
                decoded_batch, 
                scores_list
                )
        else:
            return SpeakerOutput(consistent_utterances)
    
    def score_utterances(self, contexts, targets, next_utterances):
        assert len(targets) == len(contexts), "Number of targets and contexts must be the same"
        inputs = ['' for i in range(len(targets))]
        if self.use_program:
            inputs = [inp + f"{self.program_special_token}{target}" for inp, target in zip(inputs, targets)]
        if self.use_context:
            inputs = [inp + f"{self.utterances_special_token}{self.utterances_to_string(context)}" for inp, context in zip(inputs, contexts)]

        context_tokens = self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [f"<extra_id_{len(ctx)}>" if self.idx else self.utterances_special_token for ctx in contexts]

        utterances_to_string_noidx = get_utterance_processing_functions(self.label_pos, False)[0]

        next_utterance_tokens = self.tokenizer([f"{start}{utterances_to_string_noidx([u])}" for start, u in zip(decoder_start_tokens, next_utterances)], return_tensors="pt").to(self.device)
        outputs = self.model(input_ids=context_tokens.input_ids, decoder_input_ids=next_utterance_tokens.input_ids, return_dict=True)
        
        logprobs = torch.gather(F.log_softmax(outputs.logits, dim=-1), 2, next_utterance_tokens.input_ids[:, 1:, None]).squeeze(-1)
        
        logprobs.masked_fill_(next_utterance_tokens.input_ids[:, 1:] == 0, 0)

        scores = logprobs.sum(-1)
        
        return scores.tolist()

    
    def parse_speaker_context(self, context):
        program, utterances = context.split(self.utterances_special_token)
        program = program.split(self.program_special_token)[1:]
        utterances = self.string_to_utterances(utterances)
        return program, utterances
    
    def write_formatted_training_data(self, programs, contexts, output_file, shuffle=True, len_filter=None):
        utterances_to_string_no_idx, _ = get_utterance_processing_functions(self.label_pos, False)
        write_buffer = list()

        for prog, ctx in zip(programs, contexts):
            for i in range(len(ctx) - 1):
                ctx_past = ctx[:i]
                if len_filter is not None and len(ctx_past) != len_filter:
                    continue
                u = ctx[i]
                decoder_start_token = f"<extra_id_{len(ctx_past)}>" if self.idx else self.utterances_special_token
                write_buffer.append((
                    f"{self.program_special_token}{prog}{self.utterances_special_token}{self.utterances_to_string(ctx_past)}", decoder_start_token, utterances_to_string_no_idx([u])
                    ))
        
        if shuffle:
            random.shuffle(write_buffer)

        with open(output_file, 'w') as f:
            for prog, decoder_start_token, u in write_buffer:
                f.write(f"{prog}\t{decoder_start_token}\t{u}\n")
        
        return len(write_buffer)

class SpeakerFromPrograms(Agent):
    def __init__(self, 
        model_path,
        label,
        n_distractors,
        trainable, 
        gen_config, 
        save_path=None, 
        inference_batch_size=4,
        device="cuda:0", 
        step=0, 
        name="",
        training_args=None,
        program_special_token=PROGRAM_SPECIAL_TOKEN,
        utterances_special_token=UTTERANCES_SPECIAL_TOKEN,
        gt_program_special_token=GT_PROGRAM_SPECIAL_TOKEN,
    ):
        super().__init__(model_path, trainable, save_path, gen_config, inference_batch_size, device, step, name, training_args)
        self.label_pos = "suffix"
        self.idx = False
        self.utterances_to_string, self.string_to_utterances = get_utterance_processing_functions("suffix", False)
        self.program_special_token = program_special_token
        self.utterances_special_token = utterances_special_token
        self.gt_program_special_token = gt_program_special_token
        self.label = label
        self.n_distractors = n_distractors
    
    def generate(self, targets, distractors, return_scores=False):
        assert len(targets) == len(distractors), "Number of targets and contexts must be the same"
        context_programs = [[f"{self.program_special_token}{p}" for p in d[:self.n_distractors]] + [f"{self.gt_program_special_token}{t}"] for t, d in zip(targets, distractors)]
        for ctx in context_programs:
            random.shuffle(ctx)

        contexts = [''.join(ctx) for ctx in context_programs]

        context_tokens = self.tokenizer(contexts, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [self.utterances_special_token for ctx in contexts]
        decoder_start_token_ids = self.tokenizer(decoder_start_tokens, return_tensors="pt", add_special_tokens=False).to(self.device)
        outputs = self.model.generate(
            inputs=context_tokens.input_ids, 
            decoder_input_ids=decoder_start_token_ids.input_ids,
            generation_config=self.gen_config,
            return_dict_in_generate=True,
            output_scores=True
            )

        decoded_batch = byt5_decode_batch(
            outputs.sequences.reshape((len(contexts), -1, outputs.sequences.shape[-1])).tolist(), 
            skip_special_tokens=True, 
            skip_position_token=True
            )

        consistent_utterances = list()
        idxs = list()
        for target, ctx, decoded in zip(targets, contexts, decoded_batch):
            cu = list()
            idx = list()
            for i, utterance in enumerate(decoded):
                if self.label:
                    u = self.string_to_utterances(utterance)
                else:
                    u = [(utterance, '+' if consistent(target, [(utterance, '+')]) else '-')]

                if u is None or len(u) == 0:
                    continue

                s, label = u[0]
                if len(s) > 25:
                    continue

                if not label in ['+', '-']:
                    continue

                if ' ' in s or '\t' in s or '\n' in s:
                    continue

                if consistent(target, u):
                    cu.append(u[0])
                    idx.append(i)
            
            consistent_utterances.append(cu)
            idxs.append(idx)
        
        logprobs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
        gen_probs = torch.gather(logprobs, 2, outputs.sequences[:, 1:, None]).squeeze(-1)
        gen_probs.masked_fill_(gen_probs.isinf(), 0)
        scores = gen_probs.sum(-1)
        n_decoded = scores.shape[0]
        n_seq = n_decoded // len(contexts)
        scores = scores.reshape((len(contexts), n_seq))
        scores_list = scores.tolist()

        if return_scores:
            return SpeakerOutput(
                consistent_utterances,
                idxs, 
                decoded_batch, 
                scores_list
                )
        else:
            return SpeakerOutput(consistent_utterances)
    
    def score_utterances(self, distractors, targets, next_utterances):
        assert len(targets) == len(distractors), "Number of targets and contexts must be the same"
        context_programs = [[f"{self.program_special_token}{p}" for p in d] + [f"{self.gt_program_special_token}{t}"] for t, d in zip(targets, distractors)]
        for ctx in context_programs:
            random.shuffle(ctx)

        contexts = [''.join(ctx) for ctx in context_programs]

        context_tokens = self.tokenizer(contexts, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [self.utterances_special_token for ctx in contexts]

        utterances_to_string_noidx = get_utterance_processing_functions(self.label_pos, False)[0]

        next_utterance_tokens = self.tokenizer([f"{start}{utterances_to_string_noidx([u])}" for start, u in zip(decoder_start_tokens, next_utterances)], return_tensors="pt").to(self.device)
        outputs = self.model(input_ids=context_tokens.input_ids, decoder_input_ids=next_utterance_tokens.input_ids, return_dict=True)
        
        logprobs = torch.gather(F.log_softmax(outputs.logits, dim=-1), 2, next_utterance_tokens.input_ids[:, 1:, None]).squeeze(-1)
        
        logprobs.masked_fill_(next_utterance_tokens.input_ids[:, 1:] == 0, 0)

        scores = logprobs.sum(-1)
        
        return scores.tolist()

    def parse_speaker_context(self, context):
        program, utterances = context.split(self.utterances_special_token)
        program = program.split(self.program_special_token)[1:]
        utterances = self.string_to_utterances(utterances)
        return program, utterances
    
    def write_formatted_training_data(self, programs, distractors, utterances, output_file, shuffle=True, len_filter=None):
        utterances_to_string_no_idx, _ = get_utterance_processing_functions(self.label_pos, False)
        write_buffer = list()

        for prog, ctx, u in zip(programs, distractors, utterances):
            decoder_start_token = self.utterances_special_token
            context_programs = [f"{self.program_special_token}{p}" for p in list(ctx)[:self.n_distractors]] + [f"{self.gt_program_special_token}{prog}"]
            random.shuffle(context_programs)

            # import ipdb; ipdb.set_trace()
            context = ''.join(context_programs)
            write_buffer.append((
                context, decoder_start_token, utterances_to_string_no_idx([u])
                ))
        
        if shuffle:
            random.shuffle(write_buffer)

        with open(output_file, 'w') as f:
            for prog, decoder_start_token, u in write_buffer:
                f.write(f"{prog}\t{decoder_start_token}\t{u}\n")
        
        return len(write_buffer)

class InferencePragmaticListener:
    def __init__(self, speaker, listener):
        self.speaker = speaker
        self.listener = listener

    def generate_utterances(self, programs, contexts):
        batch_size = self.speaker.inference_batch_size
        candidates = list()
        outputs = list()

        for i in range(0, len(programs), batch_size):
            speaker_outputs = self.speaker.generate(programs[i:i + batch_size], contexts[i:i + batch_size], return_scores=True)
            candidates.extend(speaker_outputs.utterances[0])
            outputs.append(speaker_outputs)
        
        return candidates, outputs

    def generate_hypotheses(self, context):
        return self.listener.synthesize(context, return_scores=True)

    def choose_next_utterance(self, hs, us, next_utterance):
        if len(hs) == 0 or len(us) == 0:
            return [], np.array([])

        hs_indexed = list(sorted(hs))
        us_indexed = list(sorted(us)) + [next_utterance]
        matrix = np.zeros((len(hs_indexed), len(us_indexed)))

        for i, d in enumerate(hs_indexed):
            for j, u in enumerate(us_indexed):
                try:
                    if consistent(d, [u]):
                        matrix[i, j] = 1
                except ValueError:
                    import ipdb; ipdb.set_trace()

        P_L0 = np.divide(
            np.ones(matrix.shape), 
            matrix.sum(axis=0, keepdims=True), 
            out=np.zeros_like(matrix), 
            where=(matrix != 0)
            )
        
        P_S1 = np.divide(
            P_L0, 
            P_L0.sum(axis=1, keepdims=True), 
            out=np.zeros_like(P_L0), 
            where=(P_L0 != 0)
            )
        
        P_L1 = np.divide(
            P_S1, 
            P_S1.sum(axis=0, keepdims=True), 
            out=np.zeros_like(P_S1), 
            where=(P_S1 != 0)
            )

        return (hs_indexed, P_L1[:, -1])
    
    def synthesize(self, context, return_scores=True):
        assert len(context) == 1, "only batch size 1 is supported"
        assert len(context[0]) >= 1, "context must contain at least one utterance"
        hs = self.generate_hypotheses(context).programs[0]
        ctx = [context[0][:-1] for h in hs]
        
        us, _ = self.generate_utterances(hs, [context[0][:-1] for h in hs])
        programs, scores = self.choose_next_utterance(hs, us, context[0][-1])
        return ListenerOutput([programs], [list(range(len(programs)))], [programs], [scores.tolist()])

class JointMoESpeaker(Speaker):
    def __init__(self, 
        model_path,
        trainable, 
        save_path, 
        gen_config, 
        inference_batch_size,
        label_pos,
        idx,
        device="cuda:0", 
        step=0, 
        name="",
        training_args=None,
        use_program=True,
        use_context=True,
        program_special_token=PROGRAM_SPECIAL_TOKEN,
        utterances_special_token=UTTERANCES_SPECIAL_TOKEN,
        type="moe"
    ):
        super().__init__(model_path,
        trainable, 
        save_path, 
        gen_config, 
        inference_batch_size,
        label_pos,
        idx,
        device, 
        step, 
        name,
        training_args,
        use_program,
        use_context,
        program_special_token,
        utterances_special_token)
    
    def generate(self, targets, contexts, return_scores=False):
        assert len(targets) == len(contexts), "Number of targets and contexts must be the same"

        target_inputs = [f"{self.program_special_token}{target}" for target in targets]
        context_inputs = [f"{self.utterances_special_token}{self.utterances_to_string(context)}" for context in contexts]

        context_tokens = self.tokenizer(context_inputs, padding=True, return_tensors="pt").to(self.model.device)
        target_tokens = self.tokenizer(target_inputs, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [f"<extra_id_{len(ctx)}>" if self.idx else self.utterances_special_token for ctx in contexts]
        decoder_start_token_ids = self.tokenizer(decoder_start_tokens, return_tensors="pt", add_special_tokens=False).to(self.device)
        logits_processor = [MoELogitsProcessor(self.model, AdditiveMixture([0.5, 0.5]), target_tokens)]
        outputs = self.model.generate(
            inputs=context_tokens.input_ids, 
            decoder_input_ids=decoder_start_token_ids.input_ids,
            generation_config=self.gen_config,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True
            )

        decoded_batch = byt5_decode_batch(
            outputs.sequences.reshape((len(contexts), -1, outputs.sequences.shape[-1])).tolist(), 
            skip_special_tokens=True, 
            skip_position_token=True
            )

        consistent_utterances = list()
        idxs = list()
        for target, decoded in zip(targets, decoded_batch):
            cu = list()
            idx = list()
            for i, utterance in enumerate(decoded):
                u = self.string_to_utterances(utterance)
                if u is None or len(u) == 0:
                    continue

                s, label = u[0]
                if len(s) > 25:
                    continue

                if not label in ['+', '-']:
                    continue

                if consistent(target, u):
                    cu.append(u[0])
                    idx.append(i)
            
            consistent_utterances.append(cu)
            idxs.append(idx)

        if return_scores:
            return SpeakerOutput(
                consistent_utterances, 
                [[scores[i] for i in idx] for scores, idx in zip(outputs.scores.tolist(), idxs)], 
                idxs, 
                decoded_batch, 
                outputs.scores.tolist()
                )
        else:
            return SpeakerOutput(consistent_utterances)
    
    def write_formatted_training_data(self, programs, contexts, output_file, incremental=True):
        with open(output_file, 'w') as f:
            for prog, ctx in zip(programs, contexts):
                if incremental:
                    end_points = range(1, len(prog) + 1)
                else:
                    end_points = [len(prog)]
                
                for end_point in end_points:
                    *ctx_past, u = ctx[:end_point]
                    decoder_start_token = f"<extra_id_{len(ctx_past)}>" if self.idx else self.utterances_special_token
                    f.write(f"{self.program_special_token}{prog}\t{decoder_start_token}\t{self.utterances_to_string([u])}\n")
                    f.write(f"{self.utterances_special_token}{self.utterances_to_string(ctx_past)}\t{decoder_start_token}\t{self.utterances_to_string([u])}\n")

class MoESpeaker(Speaker):
    def __init__(self, 
        grounded_speaker_model_path,
        utterance_lm_model_path,
        trainable, 
        save_path, 
        gen_config, 
        inference_batch_size,
        label_pos,
        idx,
        device="cuda:0", 
        step=0, 
        name="",
        training_args=None,
        use_program=True,
        use_context=True,
        program_special_token=PROGRAM_SPECIAL_TOKEN,
        utterances_special_token=UTTERANCES_SPECIAL_TOKEN,
    ):
        super().__init__(grounded_speaker_model_path,
        trainable, 
        save_path, 
        gen_config, 
        inference_batch_size,
        label_pos,
        idx,
        device, 
        step, 
        name,
        training_args,
        use_program,
        use_context,
        program_special_token,
        utterances_special_token)
        self.utterance_lm_model = AutoModelForSeq2SeqLM.from_pretrained(utterance_lm_model_path).to(device)

    def generate(self, targets, contexts, return_scores=False):
        assert len(targets) == len(contexts), "Number of targets and contexts must be the same"

        target_inputs = [f"{self.program_special_token}{target}" for target in targets]
        context_inputs = [f"{self.utterances_special_token}{self.utterances_to_string(context)}" for context in contexts]

        context_tokens = self.tokenizer(context_inputs, padding=True, return_tensors="pt").to(self.model.device)
        target_tokens = self.tokenizer(target_inputs, padding=True, return_tensors="pt").to(self.model.device)

        decoder_start_tokens = [f"<extra_id_{len(ctx)}>" if self.idx else self.utterances_special_token for ctx in contexts]
        decoder_start_token_ids = self.tokenizer(decoder_start_tokens, return_tensors="pt", add_special_tokens=False).to(self.device)
        logits_processor = [MoELogitsProcessor(self.utterance_lm_model, AdditiveMixture([0.5, 0.5]), target_tokens)]
        outputs = self.model.generate(
            inputs=context_tokens.input_ids, 
            decoder_input_ids=decoder_start_token_ids.input_ids,
            generation_config=self.gen_config,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True
            )

        decoded_batch = byt5_decode_batch(
            outputs.sequences.reshape((len(contexts), -1, outputs.sequences.shape[-1])).tolist(), 
            skip_special_tokens=True, 
            skip_position_token=True
            )

        consistent_utterances = list()
        idxs = list()
        for target, decoded in zip(targets, decoded_batch):
            cu = list()
            idx = list()
            for i, utterance in enumerate(decoded):
                u = self.string_to_utterances(utterance)
                if u is None:
                    continue

                s, label = u[0]
                if len(s) > 25:
                    continue
                if not label in ['+', '-']:
                    continue

                if consistent(target, u):
                    cu.append(u[0])
                    idx.append(i)
            
            consistent_utterances.append(cu)
            idxs.append(idx)

        if return_scores:
            return SpeakerOutput(
                consistent_utterances, 
                [[scores[i] for i in idx] for scores, idx in zip(outputs.scores.tolist(), idxs)], 
                idxs, 
                decoded_batch, 
                outputs.scores.tolist()
                )
        else:
            return SpeakerOutput(consistent_utterances)
    
    def write_formatted_training_data(self, programs, contexts, output_file, incremental=True):
        with open(output_file, 'w') as f:
            for prog, ctx in zip(programs, contexts):
                if incremental:
                    end_points = range(1, len(prog) + 1)
                else:
                    end_points = [len(prog)]
                
                for end_point in end_points:
                    *ctx_past, u = ctx[:end_point]
                    decoder_start_token = f"<extra_id_{len(ctx_past)}>" if self.idx else self.utterances_special_token
                    f.write(f"{self.program_special_token}{prog}\t{decoder_start_token}\t{self.utterances_to_string([u])}\n")
        
        with open(output_file, 'w') as f:
            for prog, ctx in zip(programs, contexts):
                if incremental:
                    end_points = range(1, len(prog) + 1)
                else:
                    end_points = [len(prog)]
                
                for end_point in end_points:
                    *ctx_past, u = ctx[:end_point]
                    decoder_start_token = f"<extra_id_{len(ctx_past)}>" if self.idx else self.utterances_special_token
                    f.write(f"{self.utterances_special_token}{self.utterances_to_string(ctx_past)}\t{decoder_start_token}\t{self.utterances_to_string([u])}\n")