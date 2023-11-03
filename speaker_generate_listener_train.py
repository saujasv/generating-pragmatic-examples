from pragmatic_speaker import generate_hypotheses, generate_utterances, choose_next_utterance
from agents import Listener, Speaker
import json
from pathlib import Path
from tqdm import tqdm
import dataclasses
import os
import wandb

def main(config):
    listener = Listener(**config['listeners'][0])
    speaker = Speaker(**config['speakers'][0])
    with open(config["user_validation_set"]) as f:
        user_validation_set = json.load(f)

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

    with open(config["train_programs"]) as f:
        train_programs = [line.strip('\n') for line in f]

    train_contexts = [list() for _ in train_programs]

    for i in tqdm(range(config["spec_len"]), desc="Spec element"):
        utterance_candidates = [set() for _ in train_programs]
        speaker.reload_model()
        utterance_candidates_speaker, speaker_outputs = generate_utterances(speaker, train_programs, train_contexts)
        for candidates, speaker_candidates in zip(
            utterance_candidates,
            utterance_candidates_speaker
            ):
            candidates.update(speaker_candidates)
        
        hypothesis_candidates = [set() for _ in train_programs]
        listener.reload_model()
        hypothesis_candidates_listener, listener_outputs = generate_hypotheses(listener, train_contexts)
        for candidates, listener_candidates in zip(
            hypothesis_candidates,
            hypothesis_candidates_listener
            ):
            candidates.update(listener_candidates)
        
        next_utterances = choose_next_utterance(hypothesis_candidates, utterance_candidates)
        for ctx, u in zip(train_contexts, next_utterances):
            if not u is None:
                ctx.append(u)
        
        with open(Path(config["working_directory"]) / f"turn-{i}-training-logs.json", 'w') as f:
            logs = []
            for prog, ctx, hyps, utts, next_utt in zip(
                train_programs, train_contexts, 
                hypothesis_candidates, utterance_candidates, 
                next_utterances):
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

    listener.write_formatted_training_data(
        train_programs,
        train_contexts,
        Path(config["working_directory"]) / f"train-contexts-{listener.name}.tsv"
        )

    train_set_paths = [
                os.path.join(
                    config["working_directory"], 
                    f"train-contexts-{listener.name}.tsv"
                )
            ]

    listener.update_model(
        {
            "train": train_set_paths,
            "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv")
        },
        str(Path(listener.save_path)), save_every=250
    )
    
    loss = listener.evaluate_loss({
            "user_validation": str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}.tsv"),
            **{
                f"user_validation_{j}": str(Path(config["working_directory"]) / f"user-validation-contexts-{listener.name}-{j}.tsv")
                for j in range(1, 9)
            }
        })

    wandb.log({
        f"outer_loop/{listener.name}_user_validation_loss": loss["user_validation"], 
        **{
            f"outer_loop/{listener.name}_user_validation_speclen={j}": loss[f"user_validation_{j}"]
            for j in range(1, 9)
        },
        f"{listener.name}_step": listener.step
        })

if __name__ == "__main__":
    import sys

    configs = [
        {
            "listener_path": "/scratch/svadugur/listener-250-continual-round1",
            "speaker_path": "/scratch/svadugur/working-speaker0=250-listener0=250-ntpr=1024-nr=10-speclen=5-lr=5e-05/speaker/round-1",
            "lr": 5e-5
        },
        {
            "listener_path": "/scratch/svadugur/listener-250-continual-round4",
            "speaker_path": "/scratch/svadugur/working-speaker0=250-listener0=250-ntpr=1024-nr=10-speclen=5-lr=5e-05/speaker/round-4",
            "lr": 5e-5
        },
        {
            "listener_path": "/scratch/svadugur/listener-250-continual-round8",
            "speaker_path": "/scratch/svadugur/working-speaker0=250-listener0=250-ntpr=1024-nr=10-speclen=5-lr=5e-05/speaker/round-8",
            "lr": 5e-5
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round1-lr=5e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-1",
            "lr": 5e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round4-lr=5e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-4",
            "lr": 5e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round8-lr=5e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-8",
            "lr": 5e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round1-lr=1e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-1",
            "lr": 1e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round4-lr=1e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-4",
            "lr": 1e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round8-lr=1e-5",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-8",
            "lr": 1e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-2000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round4-lr=5e-5-5k",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-4",
            "lr": 5e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-5000.txt"
        },
        {
            "listener_path": "/scratch/svadugur/listener-500-restart-round4-lr=1e-5-5k",
            "speaker_path": "/scratch/svadugur/working-speaker0=500-listener0=500-ntpr=1024-nr=10-speclen=5-lr=5e-05-init-all/speaker/round-4",
            "lr": 1e-5,
            "train_programs": "data/programs-corrected/listener-pragmatic-target-programs-5000.txt"
        },
    ]

    config = configs[int(sys.argv[1])]
    
    working_dir = f"working-listener-{int(sys.argv[1])}"
    args = {
        "listeners": [
            { 
                "model_path": "pragmatic-programs/listener-suffix-idx-300k",
                "trainable": True, 
                "save_path": config['listener_path'], 
                "gen_config": {
                    "do_sample": True,
                    "num_beams": 1,
                    "num_return_sequences": 500,
                    "top_p": 0.9,
                    "temperature": 1,
                    "max_new_tokens": 128
                }, 
                "label_pos": "suffix",
                "idx": True,
                "device": "cuda:0",
                "name": "listener0",
                "program_special_token": "<extra_id_124>",
                "utterances_special_token": "<extra_id_123>",
                "inference_batch_size": 1,
                "training_args": {
                    "train_batch_size": 32,
                    "eval_batch_size": 32,
                    "decoding_batch_size": 32,
                    "lr_scheduler_type": "constant",
                    "num_warmup_steps": 0,
                    "early_stopping_patience": 3,
                    "num_epochs_per_round": 1,
                    "learning_rate": config["lr"],
                    "validation_steps": 16
                },
                "inference_batch_size": 1,
            }
        ],
        "speakers": [
            {
                "type": "std",
                "model_path": config["speaker_path"],
                "trainable": False,
                "save_path": '',
                "gen_config": {
                    "do_sample": True,
                    "num_beams": 1,
                    "num_return_sequences": 500,
                    "max_new_tokens": 128
                }, 
                "label_pos": "prefix",
                "idx": True,
                "device": "cuda:0",
                "name": "speaker1",
                "program_special_token": "<extra_id_124>",
                "utterances_special_token": "<extra_id_123>",
                "inference_batch_size": 1,
            }
        ],
        "train_programs": config['train_programs'],
        "working_directory": f"/scratch/svadugur/{working_dir}",
        "spec_len": 5,
        "user_validation_set": "data/user_validation_set_round0.json",
        "inner_loop_validation_set": "user_validation"
    }

    wandb.init(project="pragmatic-regex", config=config)

    os.makedirs(os.path.join(
        args["working_directory"]
    ), exist_ok=True)

    os.makedirs(os.path.join(
        config["listener_path"]
    ), exist_ok=True)

    print(config)

    main(args)