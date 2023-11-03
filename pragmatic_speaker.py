from collections import defaultdict
from utils import consistent
import os
from tqdm import tqdm
# import wandb
import numpy as np
from agents import Listener, Speaker, JointMoESpeaker
from pathlib import Path
import logging
import functools
import time
from tqdm import tqdm
import dataclasses
import json
from multiprocessing import Pool
import itertools

def unchain(L, counts):
    unchained_L = []
    start = 0
    for c in counts:
        unchained_L.append(L[start:start + c])
        start += c
    
    return unchained_L

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"{func.__name__!r} ran for {run_time:.4f}s")
        return value
    return wrapper_timer

# @timer
def generate_utterances(speaker, programs, contexts):
    batch_size = speaker.inference_batch_size
    candidates = list()
    outputs = list()

    for i in tqdm(range(0, len(programs), batch_size), desc="Generating utterances"):
        speaker_outputs = speaker.generate(programs[i:i + batch_size], contexts[i:i + batch_size], return_scores=True)
        candidates.extend(speaker_outputs.utterances)
        outputs.append(speaker_outputs)
    
    return candidates, outputs

# @timer
def generate_hypotheses(listener, contexts, enforce_consistency=True):
    batch_size = listener.inference_batch_size
    candidates = list()
    outputs = list()

    for i in tqdm(range(0, len(contexts), batch_size), desc="Generating hypotheses"):
        listener_outputs = listener.synthesize(contexts[i:i + batch_size], return_scores=True, enforce_consistency=enforce_consistency)
        candidates.extend(listener_outputs.programs)
        outputs.append(listener_outputs)
    
    return candidates, outputs

# @timer
def choose_next_utterance(hypothesis_candidates, utterance_candidates):
    next_utterances = list()
    for (hs, us) in tqdm(zip(
        hypothesis_candidates, 
        utterance_candidates
        ), desc="Choosing next utterance"):
        if len(hs) == 0:
            if len(us) == 0:
                next_utterances.append(None)
            else:
                next_utterances.append(list(us)[0])
            continue
        elif len(us) == 0:
            next_utterances.append(None)
            continue

        hs_indexed = list(sorted(hs))
        us_indexed = list(sorted(us))
        matrix = np.zeros((len(hs) + 1, len(us)))

        # Last row is the target program, which is guaranteed to be consistent
        matrix[-1, :] = 1
        for i, d in enumerate(hs_indexed):
            for j, u in enumerate(us_indexed):
                if consistent(d, [u]):
                    matrix[i, j] = 1

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
        
        u_star_idx = P_S1[-1].argmax()
        next_utterances.append(us_indexed[u_star_idx])
    
    return next_utterances

def main(config):
    os.makedirs(os.path.join(
        config["working_directory"]
    ), exist_ok=True)

    # Load listener models
    listeners = list()
    for listener_config in config["listeners"]:
        listeners.append(Listener(**listener_config, resume=config.get("resume", None)))
        print(f"Loaded listener {listeners[-1].name} onto {listeners[-1].model.device}")

    # Load speaker models
    speakers = list()
    for speaker_config in config["speakers"]:
        if speaker_config["type"] == "moe":
            speakers.append(JointMoESpeaker(**speaker_config))
        else:
            speakers.append(Speaker(**speaker_config, resume=config.get("resume", None)))
        print(f"Loaded speaker {speakers[-1].name} onto {speakers[-1].model.device}")

    # Load user validation program-spec pairs
    with open(config["user_validation_set"]) as f:
        user_validation_set = json.load(f)

    for speaker in speakers:
        speaker.write_formatted_training_data(
            [p for p, s in user_validation_set],
            [s for p, s in user_validation_set],
            Path(config["working_directory"]) / f"user-validation-contexts-{speaker.name}.tsv"
        )

        for i in range(1, 9):
            speaker.write_formatted_training_data(
                [p for p, s in user_validation_set],
                [s for p, s in user_validation_set],
                Path(config["working_directory"]) / f"user-validation-contexts-{speaker.name}-{i}.tsv",
                len_filter=i
            )

        loss = speaker.evaluate_loss(
            {"user_validation": [
                str(Path(config["working_directory"]) / f"user-validation-contexts-{speaker.name}.tsv")
                ]})

        # wandb.log({f"outer_loop/{speaker.name}_user_validation_loss": loss["user_validation"], f"{speaker.name}_step": speaker.step})
    
    for listener in listeners:
        listener.write_formatted_training_data(
            [p for p, s in user_validation_set],
            [s for p, s in user_validation_set],
            Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv"
        )

        for i in range(1, 9):
            listener.write_formatted_training_data(
                [p for p, s in user_validation_set],
                [s for p, s in user_validation_set],
                Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}-{i}.tsv",
                len_filter=i
            )

        loss = listener.evaluate_loss(
            {"user_validation": [
                str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv")
                ]})

        # wandb.log({f"outer_loop/{listener.name}_user_validation_loss": loss["user_validation"], f"{listener.name}_step": listener.step})

    with open(config["path_to_targets"]) as f:
        program_pool = [line.strip('\n') for line in f]
        train_program_pool = program_pool[:config["num_rounds"] * config["num_train_programs_per_round"]]
        validation_program_pool = program_pool[config["num_rounds"] * config["num_train_programs_per_round"]:]

    start_round_idx = 0 if "resume" not in config else config["resume"]
    for rd in tqdm(range(start_round_idx, config["num_rounds"]), desc="Round"):
        ntpr = config["num_train_programs_per_round"]

        train_programs = train_program_pool[rd * ntpr:(rd + 1) * ntpr]
        train_contexts = [list() for _ in train_programs]

        hypothesis_candidates = [set() for _ in train_programs]
        utterance_candidates = [set() for _ in train_programs]
        for i in tqdm(range(config["spec_len"]), desc="Spec element"):            
            if not config["accumulate_hypotheses"]:
                hypothesis_candidates = [set() for _ in train_programs]
            for listener in listeners:
                listener.reload_model()
                hypothesis_candidates_listener, listener_outputs = generate_hypotheses(listener, train_contexts, config["enforce_consistency"])
                for candidates, listener_candidates in zip(
                    hypothesis_candidates,
                    hypothesis_candidates_listener
                    ):
                    candidates.update(listener_candidates)

            if config["utterance_from_gt"]:
                if not config["accumulate_utterances"]:
                    utterance_candidates = [set() for _ in train_programs]
                for speaker in speakers:
                    speaker.reload_model()
                    utterance_candidates_speaker, speaker_outputs = generate_utterances(speaker, train_programs, train_contexts)
                    for candidates, speaker_candidates in zip(
                        utterance_candidates,
                        utterance_candidates_speaker
                        ):
                        candidates.update(speaker_candidates)
            else:
                L = [[(h, ctx) for h in hs] + [(p, ctx)] for hs, p, ctx in zip(hypothesis_candidates, train_programs, train_contexts)]
                h_cand_counts = [len(x) for x in L]
                speaker_inputs = list(itertools.chain.from_iterable(L))
                utterance_candidates_chained = [set() for _ in speaker_inputs]
                for speaker in speakers:
                    speaker.reload_model()
                    utterance_candidates_speaker, speaker_outputs = generate_utterances(speaker, [h for h, ctx in speaker_inputs], [ctx for h, ctx in speaker_inputs])
                    for candidates, speaker_candidates in zip(
                        utterance_candidates_chained,
                        utterance_candidates_speaker
                        ):
                        candidates.update(speaker_candidates)
            
                utterance_candidates = [set(itertools.chain.from_iterable(x)) for x in unchain(utterance_candidates_chained, h_cand_counts)]
            
            N = len(train_programs) // config["num_workers"]
            with Pool(config["num_workers"]) as pool:
                multiple_results = [
                    pool.apply_async(choose_next_utterance, (hypothesis_candidates[i:i + N], utterance_candidates[i:i + N])) 
                    for i in range(0, len(train_programs), N)
                    ]
                next_utterances_pl = [res.get() for res in multiple_results]
            # next_utterances = choose_next_utterance(hypothesis_candidates, utterance_candidates)

            for ctx, u in zip(train_contexts, itertools.chain.from_iterable(next_utterances_pl)):
                if not u is None:
                    ctx.append(u)
            
            with open(Path(config["working_directory"]) / f"round-{rd}-turn-{i}-training-logs.json", 'w') as f:
                logs = []
                for prog, ctx, hyps, utts, next_utt in zip(
                    train_programs, train_contexts, 
                    hypothesis_candidates, utterance_candidates, 
                    itertools.chain.from_iterable(next_utterances_pl)):
                    logs.append({
                        "program": prog,
                        "context": ctx,
                        "hypotheses": list(hyps),
                        "utterances": list(utts),
                        "next_utterance": next_utt
                    })
                
                json.dump({
                    "rsa": logs, 
                    "speaker_outputs": list(map(lambda x: dataclasses.asdict(x), speaker_outputs)),
                    "listener_outputs": list(map(lambda x: dataclasses.asdict(x), listener_outputs))
                    }, f)
        
        nvpr = config["num_validation_programs_per_round"]
        validation_programs = validation_program_pool[rd * nvpr:(rd + 1) * nvpr]
        validation_contexts = [list() for _ in validation_programs]

        for i in range(config["spec_len"]):
            hypothesis_candidates = [set() for _ in validation_programs]
            for listener in listeners:
                listener.reload_model()
                hypothesis_candidates_listener, listener_outputs = generate_hypotheses(listener, validation_contexts, config["enforce_consistency"])
                for candidates, listener_candidates in zip(
                    hypothesis_candidates,
                    hypothesis_candidates_listener
                    ):
                    candidates.update(listener_candidates)

            if config["utterance_from_gt"]:
                if not config["accumulate_utterances"]:
                    utterance_candidates = [set() for _ in train_programs]
                for speaker in speakers:
                    speaker.reload_model()
                    utterance_candidates_speaker, speaker_outputs = generate_utterances(speaker, validation_programs, validation_contexts)
                    for candidates, speaker_candidates in zip(
                        utterance_candidates,
                        utterance_candidates_speaker
                        ):
                        candidates.update(speaker_candidates)
            else:
                L = [[(h, ctx) for h in hs] + [(p, ctx)] for hs, p, ctx in zip(hypothesis_candidates, validation_programs, validation_contexts)]
                h_cand_counts = [len(x) for x in L]
                speaker_inputs = list(itertools.chain.from_iterable(L))
                utterance_candidates_chained = [set() for _ in speaker_inputs]
                for speaker in speakers:
                    speaker.reload_model()
                    utterance_candidates_speaker, speaker_outputs = generate_utterances(speaker, [h for h, ctx in speaker_inputs], [ctx for h, ctx in speaker_inputs])
                    for candidates, speaker_candidates in zip(
                        utterance_candidates_chained,
                        utterance_candidates_speaker
                        ):
                        candidates.update(speaker_candidates)
            
                utterance_candidates = [set(itertools.chain.from_iterable(x)) for x in unchain(utterance_candidates_chained, h_cand_counts)]

            
            N = len(validation_programs) // config["num_workers"]
            with Pool(config["num_workers"]) as pool:
                multiple_results = [
                    pool.apply_async(choose_next_utterance, (hypothesis_candidates[i:i + N], utterance_candidates[i:i + N])) 
                    for i in range(0, len(validation_programs), N)
                    ]
                next_utterances_pl = [res.get() for res in multiple_results]

            for ctx, u in zip(validation_contexts, itertools.chain.from_iterable(next_utterances_pl)):
                if not u is None:
                    ctx.append(u)
            
            with open(Path(config["working_directory"]) / f"round-{rd}-turn-{i}-validation-logs.json", 'w') as f:
                logs = []
                for prog, ctx, hyps, utts, next_utt in zip(
                    validation_programs, validation_contexts, 
                    hypothesis_candidates, utterance_candidates, 
                    itertools.chain.from_iterable(next_utterances_pl)):
                    logs.append({
                        "program": prog,
                        "context": ctx,
                        "hypotheses": list(hyps),
                        "utterances": list(utts),
                        "next_utterance": next_utt
                    })
                
                json.dump({
                    "rsa": logs, 
                    "speaker_outputs": list(map(lambda x: dataclasses.asdict(x), speaker_outputs)),
                    "listener_outputs": list(map(lambda x: dataclasses.asdict(x), listener_outputs))
                    }, f)

        logging.info(f"TRAINING SPEAKERS") 
        for idx, speaker in enumerate(speakers):
            if not speaker.trainable:
                continue

            speaker.write_formatted_training_data(
                train_programs,
                train_contexts,
                Path(config["working_directory"]) / f"round-{rd}-train-contexts-{speaker.name}.tsv"
                )
            
            speaker.write_formatted_training_data(
                validation_programs,
                validation_contexts,
                Path(config["working_directory"]) / f"round-{rd}-validation-contexts-{speaker.name}.tsv"
                )

            if config["train_on_all_rounds"]:
                training_rounds = list(range(rd + 1))
            else:
                training_rounds = [rd]

            train_set_paths = [
                        os.path.join(
                            config["working_directory"], 
                            f"round-{j}-train-contexts-{speaker.name}.tsv"
                        )
                        for j in training_rounds
                    ]

            if config["train_from_init"]:
                speaker.reload_model(from_init=True)

            speaker.update_model(
                {
                    "train": train_set_paths,
                    **{
                        f"validation_{j}": os.path.join(
                            config["working_directory"], 
                            f"round-{j}-validation-contexts-{speaker.name}.tsv"
                        )
                        for j in range(rd + 1)
                    },
                    "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{speaker.name}.tsv")
                },
                str(Path(speaker.save_path) / f"round-{rd}")
            )
            
            loss = speaker.evaluate_loss({
                    "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{speaker.name}.tsv"),
                })

            # wandb.log({
            #     f"outer_loop/{speaker.name}_user_validation_loss": loss["user_validation"], 
            #     f"{speaker.name}_step": speaker.step
            #     })

        logging.info(f"TRAINING LISTENERS") 
        for idx, listener in enumerate(listeners):
            if not listener.trainable:
                continue

            listener.write_formatted_training_data(
                train_programs,
                train_contexts,
                Path(config["working_directory"]) / f"round-{rd}-train-contexts-{listener.name}.tsv"
                )
            
            listener.write_formatted_training_data(
                validation_programs,
                validation_contexts,
                Path(config["working_directory"]) / f"round-{rd}-validation-contexts-{listener.name}.tsv"
                )

            if config["train_on_all_rounds"]:
                training_rounds = list(range(rd + 1))
            else:
                training_rounds = [rd]

            train_set_paths = [
                        os.path.join(
                            config["working_directory"], 
                            f"round-{j}-train-contexts-{listener.name}.tsv"
                        )
                        for j in training_rounds
                    ]

            if config["train_from_init"]:
                listener.reload_model(from_init=True)

            listener.update_model(
                {
                    "train": train_set_paths,
                    **{
                        f"validation_{j}": os.path.join(
                            config["working_directory"], 
                            f"round-{j}-validation-contexts-{listener.name}.tsv"
                        )
                        for j in range(rd + 1)
                    },
                    "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv")
                },
                str(Path(listener.save_path) / f"round-{rd}")
            )
            
            loss = listener.evaluate_loss({
                    "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv"),
                })

            # wandb.log({
            #     f"outer_loop/{listener.name}_user_validation_loss": loss["user_validation"], 
            #     f"{listener.name}_step": listener.step
            #     })