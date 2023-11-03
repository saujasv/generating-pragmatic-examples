from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PragmaticSpeakerArguments:
    path_to_targets: Optional[str] = field(
        metadata={
            "help": "Path to file containing a list of target programs,"
                    "one per line"}
    )
    working_directory: Optional[str] = field(
        metadata={"help": "Path to folder for intermediate files"}
    )
    num_rounds: Optional[int] = field(
        default=None, metadata={
            "help": "Number of rounds of interleaved specification generation"
                    "and training."
        }
    )
    num_train_programs_per_round: Optional[int] = field(
        default=None, metadata={
            "help": "Number programs to be included in the training set for "
                    "which a specification is generated in each round"
        }
    )
    num_validation_programs_per_round: Optional[int] = field(
        default=None, metadata={
            "help": "Number programs to be included in the validation set for "
                    "which a specification is generated in each round"
        }
    )
    spec_len: Optional[int] = field(
        default=None, metadata={
            "help": "Number of examples to specify for each program."
        }
    )
    
    listeners_eval_dataset: Optional[str] = field(
        default=None
    )
    
    speakers_eval_dataset: Optional[str] = field(
        default=None
    )

    inner_loop_val_gt: Optional[bool] = field(default=False)
    
    train_from_init: Optional[bool] = field(default=False)

    train_on_all_rounds: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments:
    train_batch_size: Optional[int] = field(
        default=32
    )
    
    eval_batch_size: Optional[int] = field(
        default=32
    )
    
    decoding_batch_size: Optional[int] = field(
        default=32
    )
    
    lr_scheduler_type: Optional[str] = field(default="constant")
    
    num_warmup_steps: Optional[int] = field(
        default=0
    )
    
    early_stopping_patience: Optional[int] = field(
        default=3
    )
    
    num_epochs_per_round: Optional[int] = field(
        default=2
    )
    
    learning_rate: Optional[float] = field(default=5e-5)

    validation_steps: Optional[int] = field(default=20)