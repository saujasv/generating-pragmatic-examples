import random
import argparse

from StructuredRegex.toolkit.template import ConcatenationField, SeperatedField
from StructuredRegex.toolkit.filters import filter_regexes
from StructuredRegex.easy_eval.streg_utils import parse_spec_to_ast

def get_programs(n_programs, concat_max_comp=5, include_cat_regexes=True, include_sep_regexes=False, jar_lib_dir="StructuredRegex/toolkit/external"):
    sampled_programs = set()
    
    assert include_cat_regexes or include_sep_regexes, "Need to include at least one type of regex"

    while len(sampled_programs) < n_programs:
        regexes = []
        if include_cat_regexes:
            cat_regexes = [ConcatenationField.generate(concat_max_comp) for _ in range(n_programs)]
            regexes = regexes + cat_regexes
        if include_sep_regexes:
            sep_regexes = [SeperatedField.generate() for _ in range(n_programs)]
            regexes = regexes + sep_regexes
        filtered = filter_regexes(regexes, jar_lib_dir=jar_lib_dir)

        for r in filtered:
            try:
                sampled_programs.add(parse_spec_to_ast(r.specification()).standard_regex())
            except:
                pass
            if len(sampled_programs) >= n_programs:
                break
    
    return sampled_programs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--n_programs", type=int)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--concat_max_comp", type=int, default=5)
    parser.add_argument("--cat", action="store_true", default=False)
    parser.add_argument("--sep", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    print("SEED =", args.seed)
    print("CONCAT MAX COMP =", args.concat_max_comp)
    print("INCLUDE CAT =", args.cat)
    print("INCLUDE SEP =", args.sep)

    programs = get_programs(args.n_programs, args.concat_max_comp, args.cat, args.sep)

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(programs) + '\n')