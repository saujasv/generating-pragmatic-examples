import random
import argparse
import re
import logging
import rstr
import string
import itertools
from tqdm import tqdm
from utils import get_utterance_processing_functions

MAX_REP = 8
VALID = string.ascii_letters + ''.join([p for p in string.punctuation if not p in ['\\', '\'', '"']]) + string.digits
PROGRAM_SPECIAL_TOKEN="<extra_id_124>"
UTTERANCES_SPECIAL_TOKEN="<extra_id_123>"
GT_PROGRAM_SPECIAL_TOKEN="<extra_id_122>"

def sample_examples(program_set, n_pos_per_regex, n_neg_per_regex, len_limit=15, logging_steps=1000):
    positive_examples = dict()
    for i, rx in enumerate(program_set): # enumerate(tqdm(program_set, desc="Positive examples")):
        pos_ex = list()
        n_tries = 0

        rx_len_lim = rx # re.sub("{([0-9]+),}", f"{{\\1,{MAX_REP}}}", rx)
        while len(pos_ex) < n_pos_per_regex:
            n_tries += 1
            if n_tries > 5 * n_pos_per_regex:
                break

            ex = rstr.xeger(rx_len_lim)
            if len(ex) == 0 or len(ex) > len_limit or ex in pos_ex or any([False if c in VALID else True for c in ex]):
                continue
            
            pos_ex.append(ex)

        positive_examples[rx] = pos_ex

        # if i % logging_steps == 0:
        #     logging.info("Positive examples sampled for %d programs", i)
    
    # logging.info("Sampled positive examples for all programs")

    negative_examples = dict()
    for i, p in enumerate(program_set): # enumerate(tqdm(program_set, desc="Negative examples")):
        prog = re.compile(p)
        negative_examples[p] = list()
        n_tries = 0
        while len(negative_examples[p]) < n_neg_per_regex:
            n_tries += 1
            if n_tries > 1000:
                break

            ex = rstr.rstr(VALID, 1, len_limit)
            if len(ex) == 0 or len(ex) > len_limit or any([False if c in VALID else True for c in ex]):
                continue
            
            if prog.fullmatch(ex):
                continue
            
            assert not prog.fullmatch(ex)
            negative_examples[p].append(ex)
    
    specs = dict()
    for p in program_set:
        if p in positive_examples and p in negative_examples:
            specs[p] = (positive_examples[p], negative_examples[p])
    
    return specs

def sample_specs(examples, n_specs_per_program=1, min_spec_len=0, max_spec_len=15):
    data = list()
    for prog, (pos_examples, neg_examples) in tqdm(examples.items(), desc="Specs"):
        for _ in range(n_specs_per_program):
            n_examples = random.randint(min_spec_len, max_spec_len)
            if len(pos_examples) + len(neg_examples) < n_examples:
                continue

            spec = random.sample([(p, '+') for p in pos_examples] + [(n, '-') for n in neg_examples], n_examples)
            for s, label in spec:
                if label == "+":
                    assert re.fullmatch(prog, s), f"{prog}, {s}, {spec}"
                elif label == "-":
                    assert not re.fullmatch(prog, s), f"{prog}, {s}, {spec}"
            data.append((spec, prog))

    return data

def sample_for_speaker_from_programs(examples, n_specs_per_program, min_spec_len, max_spec_len):
    all_programs = list(examples.keys())
    data = list()
    for prog, (pos_examples, neg_examples) in tqdm(examples.items(), desc="Specs"):
        for _ in range(n_specs_per_program):
            n_distractors = random.randint(min_spec_len, max_spec_len)
            distractor_programs = random.sample(all_programs, n_distractors)
            example = random.choice([(s, '+') for s in pos_examples] + [(s, '-') for s in neg_examples])

            context = [f"{GT_PROGRAM_SPECIAL_TOKEN}{prog}"] + [f"{PROGRAM_SPECIAL_TOKEN}{p}" for p in distractor_programs]
            random.shuffle(context)
            data.append((''.join(context), [example]))
    
    return data

def main(args):
    random.seed(args.seed)

    program_set = list()
    with open(args.programs_file) as f:
        for line in f:
            program_set.append(line.strip())
    
    examples = sample_examples(program_set, args.n_pos_per_regex, args.n_neg_per_regex, args.len_limit)

    if args.role == "speaker_from_programs":
        specs = sample_for_speaker_from_programs(examples, args.n_specs_per_program, args.min_spec_len, args.max_spec_len)
    else:
        specs = sample_specs(examples, args.n_specs_per_program, args.min_spec_len, args.max_spec_len)
    
    utterances_to_string, _ = get_utterance_processing_functions(args.label_pos, args.add_index, separator=UTTERANCES_SPECIAL_TOKEN)
    utterances_to_string_no_idx, _ = get_utterance_processing_functions(args.label_pos, False, separator=UTTERANCES_SPECIAL_TOKEN)

    if args.role == "listener":
        with open(args.dataset_file, 'w') as f:
            for element in specs:
                spec, prog = element
                if args.add_index:
                    f.write(f"{UTTERANCES_SPECIAL_TOKEN}{utterances_to_string(spec)}\t{PROGRAM_SPECIAL_TOKEN}\t{prog}\n")
                else:
                    f.write(f"{UTTERANCES_SPECIAL_TOKEN}{utterances_to_string_no_idx(spec)}\t{PROGRAM_SPECIAL_TOKEN}\t{prog}\n")
    elif args.role == "speaker":
        with open(args.dataset_file, 'w') as f:
            for element in specs:
                spec, prog = element
                if len(spec) == 0:
                    continue
                u, ctx = spec[-1], spec[:-1]
                if args.add_index:
                    f.write(f"{PROGRAM_SPECIAL_TOKEN}{prog}{UTTERANCES_SPECIAL_TOKEN}{utterances_to_string(ctx)}\t<extra_id_{len(ctx)}>\t{utterances_to_string_no_idx([u])}\n")
                else:
                    f.write(f"{PROGRAM_SPECIAL_TOKEN}{prog}{UTTERANCES_SPECIAL_TOKEN}{utterances_to_string_no_idx}\t{UTTERANCES_SPECIAL_TOKEN}\t{utterances_to_string_no_idx([u])}\n")
    elif args.role == "speaker_from_programs":
        with open(args.dataset_file, 'w') as f:
            for element in specs:
                context, utterance = element
                f.write(f"{context}\t{UTTERANCES_SPECIAL_TOKEN}\t{utterances_to_string_no_idx(utterance)}\n")
    elif args.role == "moe_speaker":
        with open(args.utterance_lm_dataset_file, 'w') as f1, open(args.grounded_speaker_dataset_file, 'w') as f2:
            for element in specs:
                spec, prog = element
                if len(spec) == 0:
                    continue
                u, ctx = spec[-1], spec[:-1]
                bos_token = f"<extra_id_{len(ctx)}>" if args.add_index else UTTERANCES_SPECIAL_TOKEN
                f1.write(f"{UTTERANCES_SPECIAL_TOKEN}{utterances_to_string(ctx)}\t{bos_token}\t{utterances_to_string_no_idx([u])}\n")
                f2.write(f"{PROGRAM_SPECIAL_TOKEN}{prog}\t{bos_token}\t{utterances_to_string_no_idx([u])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--programs_file", type=str)
    parser.add_argument("--role", type=str, choices=["speaker", "listener", "speaker_from_programs", "moe_speaker"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_spec_len", type=int, default=0)
    parser.add_argument("--max_spec_len", type=int, default=12)
    parser.add_argument("--add_index", action="store_true")
    parser.add_argument('-special', "--use_special_index_token", action="store_true")
    parser.add_argument("--label_pos", type=str, choices=["prefix", "suffix", "none"])
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--utterance_lm_dataset_file", type=str)
    parser.add_argument("--grounded_speaker_dataset_file", type=str)
    parser.add_argument("--len_limit", type=int, default=15)
    parser.add_argument("--n_pos_per_regex", type=int, default=10)
    parser.add_argument("--n_neg_per_regex", type=int, default=10)
    parser.add_argument("--n_specs_per_program", type=int, default=1)
    args = parser.parse_args()

    print(args)
    main(args)
    
