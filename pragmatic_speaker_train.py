from pragmatic_speaker import main

if __name__ == "__main__":
    import sys

    args = [
        {
            "listeners": [
                { 
                    "model_path": "pragmatic-programs/listener-suffix-idx-300k",
                    "trainable": True, 
                    "save_path": f"working-s0=250-s1=250-l0=250-l1=250-ntpr=5120-nr=1-s=10-init-all/listener", 
                    "gen_config": {
                        "do_sample": True,
                        "num_beams": 1,
                        "num_return_sequences": 250,
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
                    "training_args": {
                        "train_batch_size": 32,
                        "eval_batch_size": 32,
                        "decoding_batch_size": 32,
                        "lr_scheduler_type": "constant",
                        "num_warmup_steps": 0,
                        "early_stopping_patience": 3,
                        "num_epochs_per_round": 1,
                        "learning_rate": 5e-5,
                        "validation_steps": 16
                    },
                    "inference_batch_size": 4,
                },
                { 
                    "model_path": "pragmatic-programs/listener-suffix-idx-300k",
                    "trainable": False,
                    "gen_config": {
                        "do_sample": True,
                        "num_beams": 1,
                        "num_return_sequences": 250,
                        "top_p": 0.9,
                        "temperature": 1,
                        "max_new_tokens": 128
                    }, 
                    "label_pos": "suffix",
                    "idx": True,
                    "device": "cuda:0",
                    "name": "listener1",
                    "program_special_token": "<extra_id_124>",
                    "utterances_special_token": "<extra_id_123>",
                    "inference_batch_size": 4
                }
            ],
            "speakers": [
                {
                    "type": "std",
                    "model_path": "pragmatic-programs/speaker-prefix-idx-300k",
                    "trainable": True, 
                    "save_path": f"working-s0=250-s1=250-l0=250-l1=250-ntpr=5120-nr=1-s=10-init-all/speaker", 
                    "gen_config": {
                        "do_sample": True,
                        "num_beams": 1,
                        "num_return_sequences": 250,
                        "max_new_tokens": 128
                    }, 
                    "label_pos": "prefix",
                    "idx": True,
                    "device": "cuda:0",
                    "name": "speaker0",
                    "program_special_token": "<extra_id_124>",
                    "utterances_special_token": "<extra_id_123>",
                    "training_args": {
                        "train_batch_size": 32,
                        "eval_batch_size": 32,
                        "decoding_batch_size": 32,
                        "lr_scheduler_type": "constant",
                        "num_warmup_steps": 0,
                        "early_stopping_patience": 3,
                        "num_epochs_per_round": 1,
                        "learning_rate": 5e-5,
                        "validation_steps": 16
                    },
                    "inference_batch_size": 4,
                },
                {
                    "type": "std",
                    "model_path": "pragmatic-programs/speaker-prefix-idx-300k",
                    "trainable": False,
                    "gen_config": {
                        "do_sample": True,
                        "num_beams": 1,
                        "num_return_sequences": 250,
                        "max_new_tokens": 128
                    }, 
                    "label_pos": "prefix",
                    "idx": True,
                    "device": "cuda:0",
                    "name": "speaker1",
                    "program_special_token": "<extra_id_124>",
                    "utterances_special_token": "<extra_id_123>",
                    "inference_batch_size": 4,
                }
            ],
            "path_to_targets": f"data/programs/pragmatic-target-programs.txt",
            "working_directory": f"working-s0=250-s1=250-l0=250-l1=250-ntpr=5120-nr=1-s=10-init-all",
            "num_rounds": 10,
            "num_train_programs_per_round": 1024,
            "num_validation_programs_per_round": 32,
            "spec_len": 10,
            "user_validation_set": "data/verified_split_val_data.json",
            "inner_loop_validation_set": "user_validation",
            "train_from_init": True,
            "train_on_all_rounds": True,
            "accumulate_utterances": False,
            "accumulate_hypotheses": False,
            "enforce_consistency": True,
            "utterance_from_gt": True,
        },
]

    print(args[int(sys.argv[1])])
    main({**args[int(sys.argv[1])], "num_workers": 16})