To train literal models:
```
python literal_trainer.py --model_name_or_path google/byt5-small --cache_dir CACHE_DIR --train_file data/programs-corrected/small-pretrain/listener-train-specs-suffix-idx.tsv --validation_file data/programs-corrected/small-pretrain/listener-validation-specs-suffix-idx.tsv --num_train_epochs 1 --lr_scheduler_type linear --warmup_ratio 0.1 --learning_rate 5e-5 --do_train --do_eval --prediction_loss_only --evaluation_strategy steps --eval_steps 375 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --save_strategy steps --save_steps 375 --save_total_limit 3 --logging_steps 5 --run_name listener-specs-suffix-idx --output_dir listener-suffix-idx-300k
```

```
python literal_trainer.py --model_name_or_path google/byt5-small --cache_dir CACHE_DIR --train_file data/programs-corrected/small-pretrain/speaker-train-specs-prefix-idx.tsv --validation_file data/programs-corrected/small-pretrain/speaker-validation-specs-prefix-idx.tsv --num_train_epochs 1 --lr_scheduler_type linear --warmup_ratio 0.1 --learning_rate 5e-5 --do_train --do_eval --prediction_loss_only --evaluation_strategy steps --eval_steps 375 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --save_strategy steps --save_steps 375 --save_total_limit 3 --logging_steps 5 --run_name speaker-specs-prefix-idx --output_dir speaker-prefix-idx-300k
```

To train pragmatic model:
`python pragmatic_speaker_train.py`

To train HFT model:
```
python literal_trainer.py --model_name_or_path pragmatic-programs/listener-suffix-idx-300k --cache_dir CACHE_DIR --train_file data/verified_split_train_data_suffix_idx.tsv --validation_file data/verified_data_listener_loss_s=full.tsv --num_train_epochs 30 --lr_scheduler_type linear --warmup_ratio 0.1 --learning_rate 5e-5 --do_train --do_eval --prediction_loss_only --evaluation_strategy epoch --eval_steps 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --save_strategy epoch --save_steps 1 --logging_steps 5 --run_name hft-listener --output_dir hft-listener
```

data/full-dataset-with-verifications.json: Full collected dataset with all verification information
data/programs/annotation-pool-sub30.txt: Pool of programs for annotation
data/programs/heldout-programs-sub30.txt: Pool of programs for user study
data/programs/pragmatic-target-programs.txt: Pool of programs for training pragmatic models
data/programs/small-pretrain/listener-train-specs-suffix-idx.tsv: Literal listener training data
data/programs/small-pretrain/listener-validation-specs-suffix-idx.tsv: Literal listener validation data
data/programs/small-pretrain/speaker-train-specs-prefix-idx.tsv: Literal speaker training data
data/programs/small-pretrain/speaker-validation-specs-prefix-idx.tsv: Literal speaker validation data
data/replay-record.json: REcords of replay data
data/verified_data_listener_loss_s=full.tsv
data/verified_split_train_data_suffix_idx.tsv
data/verified_split_train_set.json