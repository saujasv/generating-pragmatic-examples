import argparse
from agents import Speaker, Listener, JointMoESpeaker
from pragmatic_speaker import generate_hypotheses, generate_utterances, choose_next_utterance
import re
import uuid
import json
from tqdm import tqdm
import random
from utils import consistent 

def main(args):
    listener = Listener(args.listener_model_path, False, None, {
        "do_sample": True,
        "max_new_tokens": 64,
        "num_beams": 1,
        "num_return_sequences": args.listener_num_return_sequences,
    }, 1, args.listener_label_pos, args.listener_idx)


    with open(args.specs) as f:
        data = json.load(f)
        programs = [p for p, _ in data]
        contexts = [spec for _, spec in data]

    prompts = [ctx[:args.prompt_len] for ctx in contexts]
    batch_size = listener.inference_batch_size
    decoded = list()
    for i in tqdm(range(0, len(contexts), batch_size), desc="Generating hypotheses"):
        listener_outputs = listener.synthesize(prompts[i:i + batch_size], return_scores=True)
        decoded.extend(listener_outputs.decoded)
    sat_options = []
    unsat_options = []
    
    for decoded_batch, context, gt in zip(decoded, prompts, programs):
        sat = []
        unsat = []
        for i, p in enumerate(decoded_batch):
            try:
                prog = re.compile(p)
            except re.error:
                continue

            if p == gt:
                continue

            cons = consistent(p, context)
            if cons and len(sat) < 2:
                sat.append(p)
            elif len(unsat) < 2:
                unsat.append(p)
        
        sat_options.append(sat)
        unsat_options.append(unsat)

    data = dict()
    for program, ctx, sat, unsat in zip(programs, contexts, sat_options, unsat_options):
        progid = str(uuid.uuid4())
        options = [{
                    "ground_truth": True,
                    "regex": program,
                    "sat": True
                }] + [
                    {
                    "ground_truth": False,
                    "regex": p,
                    "sat": True
                } for p in sat] + [
                    {
                    "ground_truth": False,
                    "regex": program,
                    "sat": False
                } for program in unsat
                ]
        
        random.shuffle(options)
        data[progid] = {
            "examples": [{"id": i, "string": s, "label": l} for i, (s, l) in enumerate(ctx)],
            "options": [{"id": i, **o} for i, o in enumerate(options)]
        }
    
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--specs", type=str, required=True)
    parser.add_argument("--listener_model_path", type=str, required=True)
    parser.add_argument("--listener_label_pos", type=str, required=True, choices=["prefix", "suffix"])
    parser.add_argument("--listener_num_return_sequences", type=int, default=500)
    parser.add_argument("--listener_idx", action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_len", type=int, default=1)

    args = parser.parse_args()
    
    main(args)
