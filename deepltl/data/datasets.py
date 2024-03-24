
# pylint: disable = line-too-long

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from deepltl.data import ltl_parser
from deepltl.data.vocabulary import LTLVocabulary, TraceVocabulary

def get_dataset_splits(dataset_name, splits, dataset_class, dataset_args, data_dir=None):
    data_dir = data_dir if data_dir is not None else os.path.join(os.path.dirname(__file__), '../../../data')
    dataset_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError('Cannot access dataset directory ' + str(self.dataset_dir))
    return [
        dataset_class(os.path.join(dataset_dir, split + '.txt'), **dataset_args)
        for split in splits
    ]


class LTLTracesDataset(Dataset):
    """Dataset that consists of pairs of a LTL formula and a satisfying trace."""

    def __init__(
        self, 
        filename,
        ltl_vocab: LTLVocabulary,
        trace_vocab: TraceVocabulary,
        max_length_formula,
        max_length_trace,
        prepend_start_token,
        tree_pos_enc,
        max_samples = None,
    ):
        """
        Expects data file to have formula\ntrace\n format
        """

        def process_pair(line_in, line_out):
            formula = ltl_parser.ltl_formula(line_in, 'network-polish')
            encoded_in = ltl_vocab.encode(formula.to_str('network-polish', spacing='all ops').split(' '))
            encoded_out = trace_vocab.encode(line_out, prepend_start_token=prepend_start_token)
            if tree_pos_enc:
                position_list = formula.binary_position_list(format='lbt', add_first=True)
                # pad to max length
                max_length = max([len(l) for l in position_list])
                padded_position_list = [l + [0] * (max_length - len(l)) for l in position_list]
                datum = torch.tensor(encoded_in), torch.tensor(encoded_out), torch.tensor(padded_position_list, dtype=torch.float32)
            else:
                datum = torch.tensor(encoded_in), torch.tensor(encoded_out)
            return datum

        pairs = []
        with open(filename, 'r') as file:  # expect formula\ntrace\n format
            for line_in in file:
                if line_in == '\n':
                    break
                line_in = line_in.strip()
                line_out = next(file).strip()  # get second line
                if max_length_formula >= 0 and len(line_in) > max_length_formula:
                    continue
                if max_length_trace >= 0 and len(line_out) > max_length_trace:
                    continue
                pairs.append((line_in, line_out))

        if max_samples is not None:
            pairs = pairs[:max_samples]

        self.data = [process_pair(*pair) for pair in tqdm(pairs, desc=os.path.basename(filename))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BooleanSatDataset(Dataset):
    def __init__(
            self,
            filename,
            formula_vocab,
            assignment_vocab,
            tree_pos_enc,
            max_samples = None,
        ):

        def process_pair(line_in, line_out):
            formula = ltl_parser.ltl_formula(line_in, 'network-polish')
            encoded_in = formula_vocab.encode(formula.to_str('network-polish', spacing='all ops').split(' '))
            encoded_out = assignment_vocab.encode(line_out)
            if tree_pos_enc:
                position_list = formula.binary_position_list(format='lbt', add_first=True)
                # pad to max length
                max_length = max([len(l) for l in position_list])
                padded_position_list = [l + [0] * (max_length - len(l)) for l in position_list]
                datum = torch.tensor(encoded_in), torch.tensor(encoded_out), torch.tensor(padded_position_list, dtype=torch.float32)
            else:
                datum = torch.tensor(encoded_in), torch.tensor(encoded_out)
            return datum

        pairs = []
        with open(filename, 'r') as file:  # expect formula\ntrace\n format
            for line_in in file:
                if line_in == '\n':
                    return
                line_in = line_in.strip()
                line_out = next(file).strip()  # get second line
                pairs.append((line_in, line_out))

        if max_samples is not None:
            pairs = pairs[:max_samples]

        self.data = [process_pair(*pair) for pair in tqdm(pairs, desc=os.path.basename(filename))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]