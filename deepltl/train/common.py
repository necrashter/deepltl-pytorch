from collections import defaultdict
import os
import json
import sys
import random
import subprocess
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from deepltl.data.vocabulary import CharacterVocabulary
from deepltl.data import ltl_parser, trace_check
from deepltl.utils.utils import tictoc_histogram


class CustomPadCollate:
    """
    A collate function that pads the input sequences to the longest sequence in the batch.
    """

    def __init__(self, d_embed_enc=None):
        """
        args:
            d_embed_enc: None if custom positional encoding is not used, embedding dimension otherwise
        """
        self.d_embed_enc = d_embed_enc

    def __call__(self, batch):
        """
        args:
            batch: list of (input, target) or (input, target, positional_encoding)

        Returns:
            (input, target) or (input, target, positional_encoding)
                input: int tensor with shape (batch_size, input_length)
                target: int tensor with shape (batch_size, target_length)
                positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        # Pad by adding zeros to the end
        max_input_length = max(map(lambda x: x[0].size(0), batch))
        xs = torch.stack([F.pad(x[0], (0, max_input_length - x[0].size(0)), "constant", 0) for x in batch], dim=0)
        max_target_length = max(map(lambda x: x[1].size(0), batch))
        ys = torch.stack([F.pad(x[1], (0, max_target_length - x[1].size(0)), "constant", 0) for x in batch], dim=0)
        if self.d_embed_enc is not None:
            pe = torch.stack([F.pad(x[2], (0, self.d_embed_enc - x[2].size(-1), 0, max_input_length - x[2].size(-2)), "constant", 0) for x in batch], dim=0)
            return xs, ys, pe
        else:
            return xs, ys


def argparser():
    parser = ArgumentParser()
    # Meta
    parser.add_argument('--run-name', default='default', help='name of this run, to better find produced data later')
    parser.add_argument('--job-dir', default='runs', help='general job directory to save produced data into')
    parser.add_argument('--data-dir', default='data', help='directory of datasets')
    parser.add_argument('--ds-name', default='ltl-35', help='Name of the dataset to use')
    do_test = parser.add_mutually_exclusive_group()
    do_test.add_argument('--train', dest='test', action='store_false', default=False, help='Run in training mode, do not perform testing; default')
    do_test.add_argument('--test', dest='test', action='store_true', default=False, help='Run in testing mode, do not train')
    parser.add_argument('--binary-path', default=None, help='Path to binaries, current: aalta')
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (default: cuda if available)")

    # Typical Hyperparameters
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial-epoch', type=int, default=0, help='used to track the epoch number correctly when resuming training')
    parser.add_argument('--training-samples', type=int, default=None)

    # Beam search
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beam-size', type=int, default=2)

    # Evaluation
    parser.add_argument('--eval-threads', type=int, help="Number of threads used while trace checking")
    parser.add_argument('--eval-timeout', type=int, default=30, help="Timeout before the trace checking for a single trace is terminated in seconds")

    return parser


def setup(binary_path, seed, **kwargs):
    # GPU stuff
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
        torch.backends.cudnn.benchmark = True
    else:
        print("Warning: CUDA not available, running on CPU.")

    if seed is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        # Not used but just to make sure
        random.seed(seed)
        np.random.seed(seed)
        print("Seed:", seed)
    else:
        print("No seed is given")
        
    # Get binaries
    filenames = ['aalta']
    if binary_path is not None:
        for filename in filenames:
            try:
                os.makedirs('bin', exist_ok=True)
                subprocess.run(['cp', os.path.join(binary_path, filename), os.path.join('bin', filename)], check=True)
            except FileExistsError:
                pass


def get_run_path(dir_name, job_dir, run_name, **kwargs):
    return os.path.join(job_dir, run_name, dir_name)

def get_latest_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None

    # Get all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pth")]

    # No checkpoint files found
    if not checkpoint_files:
        return None

    filename = max(checkpoint_files, key=lambda x: int(x.split("_", 1)[0][2:]))  # Extract epoch number

    # Return path to the latest checkpoint
    return os.path.join(checkpoint_path, filename)


def test_and_analyze_ltl(pred_fn, dataloader, torch_device, in_vocab, out_vocab, **kwargs):
    plotdir = os.path.join(kwargs['job_dir'], kwargs['run_name'], "test")
    os.makedirs(plotdir, exist_ok=True)
    # List of tuples (formula, predicted trace, target trace)
    predictions = []
    for x in tqdm(dataloader, desc="Predict"):
        x = tuple(datum.to(torch_device) for datum in x)
        if kwargs['tree_pos_enc']:
            data, label, pe = x
            pred = pred_fn(data, pe)
        else:
            data, label = x
            pred = pred_fn(data)
        for i in range(pred.shape[0]):
            label_decoded = out_vocab.decode(list(label[i, :]))
            if not label_decoded:
                label_decoded = '-'
            formula = in_vocab.decode(list(data[i, :]))
            trace = out_vocab.decode(list(pred[i, :]))
            predictions.append((formula, trace, label_decoded))

    results = trace_check.evaluate_ltl(predictions, threads=kwargs["eval_threads"], timeout=kwargs["eval_timeout"])
    with open(os.path.join(plotdir, "evaluation.json"), 'w') as f:
        f.write(json.dumps(results, indent=4))

    eval_times = {"Trace Evaluation Times": [result["time"] for result in results]}
    tictoc_histogram(eval_times, save_to=os.path.join(plotdir, "trace_times.png"), figsize=(8, 5))

    analysis = trace_check.analyze_results(results)
    res = trace_check.per_size_analysis(analysis, save_analysis=os.path.join(plotdir, "size_hist"))
    total = len(results)
    res["correct"] = res["exact match"] + res["semantically correct"]
    # For ordering headers in res
    order = defaultdict(lambda: 100)
    for i, e in enumerate(["correct", "exact match", "semantically correct", "incorrect"]):
        order[e] = i
    # Create summary string
    summary = ["EVALUATION SUMMARY"]
    for key, count in sorted(res.items(), key=lambda pair: order[pair[0]]):
        summary.append(f"{key.capitalize()}: {count}/{total}, {count / total * 100:f}%")
    summary = "\n".join(summary)

    print()
    print(summary)
    with open(os.path.join(plotdir, "summary.txt"), 'w') as f:
        f.write(summary)
        f.write("\n\nCommand Line Arguments:\n")
        f.write(" ".join(sys.argv[1:]))
        f.write("\n")


def get_ass(lst):
    if len(lst) % 2 != 0:
        raise ValueError('length of assignments not even')
    ass_it = iter(lst)
    ass_dict = {}
    for var in ass_it:
        val = next(ass_it)
        if val == 'True' or val == '1':
            ass_dict[var] = True
        elif val == 'False' or val == '0':
            ass_dict[var] = False
        else:
            raise ValueError('assignment var not True or False')
    s = [f'{var}={val}' for (var, val) in ass_dict.items()]
    return ass_dict, ' '.join(s)


def test_and_analyze_sat(pred_model, dataloader, torch_device, in_vocab, out_vocab, log_name, **kwargs):
    from deepltl.data.sat_generator import spot_to_pyaiger, is_model

    logdir = os.path.join(kwargs['job_dir'], kwargs['run_name'])
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, log_name), 'w') as log_file:
        res = {'invalid': 0, 'incorrect': 0, 'syn_correct': 0, 'sem_correct': 0}
        for x in dataloader:
            x = tuple(datum.to(torch_device) for datum in x)
            if kwargs['tree_pos_enc']:
                data, label_, pe = x
                prediction = pred_model(data, None, pe)["outputs"]
            else:
                data, label_ = x
                prediction = pred_model(data)["outputs"]
            for i in range(prediction.shape[0]):
                formula = in_vocab.decode(list(data[i, :]), as_list=True)
                pred = out_vocab.decode(list(prediction[i, :]), as_list=True)
                label = out_vocab.decode(list(label_[i, :]), as_list=True)
                formula_obj = ltl_parser.ltl_formula(''.join(formula), 'network-polish')
                formula_str = formula_obj.to_str('spot')
                _, pretty_label_ass = get_ass(label)
                try:
                    _, pretty_ass = get_ass(pred)
                except ValueError as e:
                    res['invalid'] += 1
                    msg = f"INVALID ({str(e)})\nFormula: {formula_str}\nPred:     {' '.join(pred)}\nLabel:    {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if pred == label:
                    res['syn_correct'] += 1
                    msg = f"SYNTACTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:    {pretty_label_ass}\n"
                    # log_file.write(msg)
                    continue

                # semantic checking
                formula_pyaiger = spot_to_pyaiger(formula)
                ass_pyaiger = spot_to_pyaiger(pred)
                pyaiger_ass_dict, _ = get_ass(ass_pyaiger)
                # print(f'f: {formula_pyaiger}, ass: {pyaiger_ass_dict}')
                try:
                    holds = is_model(formula_pyaiger, pyaiger_ass_dict)
                except KeyError as e:
                    res['incorrect'] += 1
                    msg = f"INCORRECT (var {str(e)} not in formula)\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if holds:
                    res['sem_correct'] += 1
                    msg = f"SEMANTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                else:
                    res['incorrect'] += 1
                    msg = f"INCORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:   {pretty_label_ass}\n"
                    log_file.write(msg)

        total = sum(res.values())
        correct = res['syn_correct'] + res['sem_correct']
        msg = (f"Correct: {correct/total*100:.1f}%, {correct} out of {total}\nSyntactically correct: {res['syn_correct']/total*100:.1f}%\nSemantically correct: {res['sem_correct']/total*100:.1f}%\n"
               f"Incorrect: {res['incorrect']/total*100:.1f}%\nInvalid: {res['invalid']/total*100:.1f}%\n")
        log_file.write(msg)
        print(msg, end='')


def log_params(job_dir, run_name, **kwargs):
    logdir = os.path.join(job_dir, run_name)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'params.txt'), 'w') as f:
        for key, val in kwargs.items():
            f.write('{}: {}\n'.format(key, val))
