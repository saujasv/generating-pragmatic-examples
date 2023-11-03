import regex as re
import random
from dataclasses import dataclass
from typing import Optional, Any, Union
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np

def consistent(rx, spec):
    # spec is in the form of (string, '+'/'-') pairs
    for s, label in spec:
        if not label in ['+', '-']:
            return None
        try:
            if re.fullmatch(rx, s, timeout=1):
                if label == '-':
                    return False
            else:
                if label == '+':
                    return False
        except re.error:
            return None
        except TimeoutError:
            return None

    return True

def decode(c):
    if c < 3:
        return f"<{c}>"
    elif c < 258:
        return chr(c - 3)
    else:
        return f"<extra_id_{c - 259}>"
        
def byt5_decode_batch(outputs, skip_special_tokens=True, skip_position_token=False):
    skipped_tokens = outputs
    if skip_special_tokens:
        skipped_tokens = [
            [[t for t in x if t >= 3] for x in beam]
            for beam in skipped_tokens
            ]
    
    if skip_position_token:
        skipped_tokens = [
            [[t for t in x if t <= 258] for x in beam] 
            for beam in skipped_tokens
            ]

    return [
        [''.join([decode(t) for t in x]) for x in beam]
        for beam in skipped_tokens
    ]

def get_preprocess_function(tokenizer):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            [' ' if x is None else x for x in examples["context"]], 
            text_target=examples["target"], 
            truncation=True
        )
        return model_inputs
    
    return preprocess_function

def get_utterance_processing_functions(label_pos, idx, separator=' '):
    if label_pos == "suffix":
        if idx:
            def utterances_to_string(spec):
                return ''.join([f"<extra_id_{i}>{s}{label}" for i, (s, label) in enumerate(spec)])
        else:
            def utterances_to_string(spec):
                return separator.join([f"{s}{label}" for s, label in spec])
    else:
        if idx:
            def utterances_to_string(spec):
                return ''.join([f"<extra_id_{i}>{label}{s}" for i, (s, label) in enumerate(spec)])
        else:
            def utterances_to_string(spec):
                return separator.join([f"{label}{s}" for s, label in spec])
    
    if label_pos == "suffix":
        if idx:
            def string_to_utterances(string):
                string = re.sub(r'<extra_id_\d+>', ' ', string)
                return [(s[:-1], s[-1]) for s in string.split(' ') if len(s) > 0]
        else:
            def string_to_utterances(string):
                return [(s[:-1], s[-1]) for s in string.split(separator) if len(s) > 0]
    else:
        if idx:
            def string_to_utterances(string):
                string = re.sub(r'<extra_id_\d+>', '', string)
                return [(s[1:], s[0]) for s in string.split(separator) if len(s) > 0]
        else:
            def string_to_utterances(string):
                return [(s[1:], s[0]) for s in string.split(separator) if len(s) > 0]
    
    return utterances_to_string, string_to_utterances

@dataclass
class DataCollatorForSeq2Seq:
    """
    Modified from https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/data/data_collator.py
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                decoder_inputs_remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["decoder_input_ids"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                    if "decoder_input_ids" in feature:
                        feature["decoder_input_ids"] = (
                            feature["decoder_input_ids"] + decoder_inputs_remainder if padding_side == "right" else decoder_inputs_remainder + feature["decoder_input_ids"]
                        )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    if "decoder_input_ids" in feature:
                        feature["decoder_input_ids"] = np.concatenate([feature["decoder_input_ids"], decoder_inputs_remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    if "decoder_input_ids" in feature:
                        feature["decoder_input_ids"] = np.concatenate([decoder_inputs_remainder, feature["decoder_input_ids"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and not "decoder_input_ids" in features
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features