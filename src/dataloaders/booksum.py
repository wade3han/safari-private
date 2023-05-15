"""Booksum dataset"""
import logging
import pickle

from torch.utils.data import Subset
from transformers import AutoTokenizer
from torch import nn

from datasets import DatasetDict, load_dataset
from src.dataloaders.base import default_data_path, SequenceDataset

MASK_IDX = 50255 # FIXME (@seungjuh) refer to ignore_index in cross_entropy in torch


class BookSum(SequenceDataset):
    _name_ = "kmfoda/booksum"

    @property
    def init_defaults(self):
        return {
            "max_length": 8192,
            "seed": 42,
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return len(self.tokenizer)

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer = self.process_dataset()
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["validation"],
            dataset["test"],
        )
        test_ids = list(range(100))  # use only 100 samples
        self.dataset_test = Subset(self.dataset_test, test_ids)

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        # lengths = torch.tensor([len(x) for x in xs])
        xs = nn.utils.rnn.pad_sequence(
            xs, padding_value=MASK_IDX, batch_first=True
        )
        ys = nn.utils.rnn.pad_sequence(
            ys, padding_value=MASK_IDX, batch_first=True
        )
        return xs, ys, {}

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"],
                              validation=dataset["validation"],
                              test=dataset["test"])

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        INSTRUCT = ' \n What is the summary of the given text? \n '

        def get_source(tokenizer, example):
            input_ids = tokenizer(example['chapter'] + INSTRUCT)['input_ids']
            target_ids = tokenizer(example['summary_text'])['input_ids']

            return input_ids + target_ids[:-1]

        def get_target(tokenizer, example):
            input_ids = tokenizer(example['chapter'] + INSTRUCT)['input_ids']
            target_ids = tokenizer(example['summary_text'])['input_ids']
            masks = [MASK_IDX] * len(input_ids)
            return masks[1:] + target_ids if not self.append_eos else masks[1:] + target_ids[:-1] + [tokenizer.eos_token_id]

        # input_ids = tokenize_maybe_eos(sources)['input_ids']

        # Account for <bos> and <eos> tokens
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        input_tokenize = lambda example: {"input_ids": get_source(tokenizer, example)[-max_length:]}
        dataset = dataset.map(
            input_tokenize,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        target_tokenize = lambda example: {"label": get_target(tokenizer, example)[-max_length:]}
        dataset = dataset.map(
            target_tokenize,
            remove_columns=["chapter", "summary_text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, cache_dir)
        return dataset, tokenizer

    def _save_to_cache(self, dataset, tokenizer, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return dataset, tokenizer

    @property
    def _cache_dir_name(self):
        return f"max_length-{self.max_length}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"
