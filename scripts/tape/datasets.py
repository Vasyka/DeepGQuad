from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from .tokenizers import TAPETokenizer
from .registry import registry

logger = logging.getLogger(__name__)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class FastaDataset(Dataset):
    """Creates a dataset from a fasta file.
    Args:
        data_file (Union[str, Path]): Path to fasta file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        from Bio import SeqIO
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        # if in_memory:
        cache = list(SeqIO.parse(str(data_file), 'fasta'))
        num_examples = len(cache)
        self._cache = cache
        # else:
            # records = SeqIO.index(str(data_file), 'fasta')
            # num_examples = len(records)
#
            # if num_examples < 10000:
                # logger.info("Reading full fasta file into memory because number of examples "
                            # "is very low. This loads data approximately 20x faster.")
                # in_memory = True
                # cache = list(records.values())
                # self._cache = cache
            # else:
                # self._records = records
                # self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        record = self._cache[index]
        # else:
            # key = self._keys[index]
            # record = self._records[key]
            # if self._in_memory:
                # self._cache[index] = record

        item = {'id': record.id,
                'primary': str(record.seq),
                'protein_length': len(record.seq)}
        return item

class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
        import json
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())

        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = self._records[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item



@registry.register_task('masked_language_modeling')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        clan = torch.LongTensor(clan)  # type: ignore
        family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels
        


