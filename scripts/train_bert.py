from typing import Union, List, Tuple, Any, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from copy import copy
import pickle as pkl
import logging
import random

from tape.datasets import LMDBDataset, FastaDataset, JSONDataset, pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from tape import ProteinBertForSequenceToSequenceClassification, ProteinResNetForMaskedLM, ProteinBertForMaskedLM


@registry.register_task('dna_language_modeling')
class DNALanguageModelingDataset(Dataset):
    """ Defines the 8-class secondary structure prediction dataset.
    Args:
        data_path (Union[str, Path]): Path to tape data directory. By default, this is
            assumed to be `./data`. Can be altered on the command line with the --data_dir
            flag.
        split (str): The specific dataset split to load often <train, valid, test>. In the
            case of secondary structure, there are three test datasets so each of these
            has a separate split flag.
        tokenizer (str): The model tokenizer to use when returning tokenized indices.
        in_memory (bool): Whether to load the entire dataset into memory or to keep
            it on disk.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'dna',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'small_{split}.fa'
        self.data = FastaDataset(data_path / data_file, in_memory=in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """ Return an item from the dataset. We've got an LMDBDataset that
            will load the raw data and return dictionaries. We have to then
            take that, load the keys that we need, tokenize and convert
            the amino acids to ids, and return the result.
        """
        item = self.data[index]
        # tokenize + convert to numpy
        token_ids = self.tokenizer.encode(item['primary'])
        # this will be the attention mask - we'll pad it out in
        # collate_fn
        input_mask = np.ones_like(token_ids)

        return token_ids, input_mask 

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """ Define a collate_fn to convert the variable length sequences into
            a batch of torch tensors. token ids and mask should be padded with
            zeros. Labels for classification should be padded with -1.
            This takes in a list of outputs from the dataset's __getitem__
            method. You can use the `pad_sequences` helper function to pad
            a list of numpy arrays.
        """
        input_ids, input_mask = tuple(zip(*batch))

        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        torch_labels = torch.from_numpy(pad_sequences(input_ids, -1))

        return {'input_ids': torch_inputs,
                'input_mask': input_mask,
                'targets': torch_labels}
                
                
@registry.register_task('dna_masked_language_modeling')
class DNAMaskedLanguageModelingDataset(Dataset):
    """ Defines the 8-class secondary structure prediction dataset.
    Args:
        data_path (Union[str, Path]): Path to tape data directory. By default, this is
            assumed to be `./data`. Can be altered on the command line with the --data_dir
            flag.
        split (str): The specific dataset split to load often <train, valid, test>. In the
            case of secondary structure, there are three test datasets so each of these
            has a separate split flag.
        tokenizer (str): The model tokenizer to use when returning tokenized indices.
        in_memory (bool): Whether to load the entire dataset into memory or to keep
            it on disk.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'dna',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'chr1_{split}.fa'
        self.data = FastaDataset(data_path / data_file, in_memory=in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """ Return an item from the dataset. We've got an LMDBDataset that
            will load the raw data and return dictionaries. We have to then
            take that, load the keys that we need, tokenize and convert
            the amino acids to ids, and return the result.
        """
        item = self.data[index]
        # tokenize + convert to numpy
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
            
        return masked_token_ids, input_mask, labels


    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """ Define a collate_fn to convert the variable length sequences into
            a batch of torch tensors. token ids and mask should be padded with
            zeros. Labels for classification should be padded with -1.
            This takes in a list of outputs from the dataset's __getitem__
            method. You can use the `pad_sequences` helper function to pad
            a list of numpy arrays.
        """
        input_ids, input_mask, lm_label_ids = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

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


@registry.register_task('g_quadruplex', num_labels=2)
class GQuadruplexDataset(Dataset):
    """ Defines the 8-class secondary structure prediction dataset.
    Args:
        data_path (Union[str, Path]): Path to tape data directory. By default, this is
            assumed to be `./data`. Can be altered on the command line with the --data_dir
            flag.
        split (str): The specific dataset split to load often <train, valid, test>. In the
            case of secondary structure, there are three test datasets so each of these
            has a separate split flag.
        tokenizer (str): The model tokenizer to use when returning tokenized indices.
        in_memory (bool): Whether to load the entire dataset into memory or to keep
            it on disk.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'dna',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'readydata_{split}.json'
        self.data = JSONDataset(data_path / data_file, in_memory=in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """ Return an item from the dataset. We've got an LMDBDataset that
            will load the raw data and return dictionaries. We have to then
            take that, load the keys that we need, tokenize and convert
            the amino acids to ids, and return the result.
        """
        item = self.data[index]
        # tokenize + convert to numpy
        token_ids = self.tokenizer.encode(item['primary'])
        # this will be the attention mask - we'll pad it out in
        # collate_fn
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['gquad'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """ Define a collate_fn to convert the variable length sequences into
            a batch of torch tensors. token ids and mask should be padded with
            zeros. Labels for classification should be padded with -1.
            This takes in a list of outputs from the dataset's __getitem__
            method. You can use the `pad_sequences` helper function to pad
            a list of numpy arrays.
        """
        input_ids, input_mask, gquad_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        gquad_label = torch.from_numpy(pad_sequences(gquad_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': gquad_label}

        return output



registry.register_task_model(
    'dna_language_modeling', 'transformer', ProteinBertForSequenceToSequenceClassification)

registry.register_task_model(
     'dna_masked_language_modeling', 'resnet', ProteinResNetForMaskedLM)
registry.register_task_model(
    'dna_masked_language_modeling','transformer',ProteinBertForMaskedLM)
    
registry.register_task_model(
    'g_quadruplex','transformer',ProteinBertForSequenceToSequenceClassification)

if __name__ == '__main__':
    from tape.main import run_train, run_train_distributed
    run_train() 