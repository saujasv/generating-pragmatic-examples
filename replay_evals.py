from agents import Listener, Speaker, InferencePragmaticListener
import json
import re
import numpy as np
import argparse
from tqdm import tqdm
import time

def main(args):
    if args.search_type == "sampling":
        print("Using sampling listener")
        listener = Listener(args.model_path, False, {"do_sample": True, "num_beams": 1, "max_new_tokens": 128, "top_p": 0.9, "temperature": 1, "num_return_sequences": args.n_samples})
    elif args.search_type == "inference":
        print("Using inference listener")
        speaker = Speaker(args.speaker_model_path, False, {"do_sample": True, "num_beams": 1, "max_new_tokens": 128, "top_p": 1, "temperature": 1, "num_return_sequences": args.n_samples})
        literal_listener = Listener(args.model_path, False, {"do_sample": True, "num_beams": 1, "max_new_tokens": 128, "top_p": 0.9, "temperature": 1, "num_return_sequences": args.n_samples})
        listener = InferencePragmaticListener(speaker, literal_listener)

    with open(args.interactions_file) as f:
        interactions_data = json.load(f)

    eval_logs = []
    for episode in tqdm(interactions_data):
        episode_log = []
        turns = episode["interaction"] if "interaction" in episode else episode["log"]
        for turn_idx, turn in enumerate(turns):
            ctx = turn["context"][:turn_idx + 1]
            start = time.time()
            outputs = listener.synthesize([ctx], return_scores=True)
            consistent_program_scores = [outputs.decoded_scores[0][i] for i in outputs.idx[0]]
            consistent_programs = [outputs.decoded[0][i] for i in outputs.idx[0]]
            sorted_programs = sorted(zip(consistent_program_scores, consistent_programs), reverse=True, key=lambda x: x[0])
            sorted_programs_dedup = sorted(set(zip(consistent_program_scores, consistent_programs)), reverse=True, key=lambda x: x[0])
            end = time.time()
            episode_log.append({
                "context": list(ctx),
                "time": end - start,
                "top_programs": sorted_programs,
                "top_programs_dedup": sorted_programs_dedup
            })
        eval_logs.append({
            "program": episode["program"],
            "log": episode_log
            })
    
    with open(args.output_file, "w") as f:
        json.dump({"replay": eval_logs}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--speaker_model_path", type=str)
    parser.add_argument("--interactions_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--search_type", type=str, choices=["sampling", 'inference'], required=True)
    args = parser.parse_args()
    print(args)
    main(args)
