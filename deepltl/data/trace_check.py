# pylint: disable=line-too-long

import sys
import argparse
import math
import multiprocessing
import time
from collections import defaultdict
from functools import reduce, partial
from tqdm import tqdm
import spot

from deepltl.data import ltl_parser
from deepltl.data import aalta_wrapper
from deepltl.data.ltl_parser import LTLTrace, LTLFormula, F_AND, F_IMLIES, F_NEXT, F_GLOBALLY, F_NOT, F_AP


def per_size_analysis(full_results, **kwargs):
    import matplotlib.pyplot as plt

    colors = {
        'syntactically correct': '#38b547',
        'exact match': '#38b547',
        'only semantically correct': '#85f67c',
        'semantically correct': '#85f67c',
        'incorrect': '#ed974d',
        'invalid': '#fd4a4a',
    }
    results = {k: v for k, v in full_results.items() if k in colors}
    order = {
        'syntactically correct': 0,
        'exact match': 0,
        'only semantically correct': 1,
        'semantically correct': 1,
        'incorrect': 2,
        'invalid': 3,
    }
    results = dict(sorted(results.items(), key=lambda pair: order[pair[0]]))

    min_size = min([min(d) if len(d) > 0 else math.inf for d in results.values()])
    max_size = max([max(d) if len(d) > 0 else 0 for d in results.values()])
    x, totals = [], []
    assert not ('total' in results)
    results_complete = {}
    for size in range(min_size, max_size + 1):
        x.append(size)
        totals.append(0)
    bottom_positions = totals.copy()

    for category, dist in results.items():  # dict with sizes to list; not all values may occur in dict
        results_complete[category] = []
        for idx, size in enumerate(range(min_size, max_size + 1)):
            value = dist[size] if size in dist else 0
            results_complete[category].append(value)
            totals[idx] += value
    results_percent = {}
    for category, dist_complete in results_complete.items():
        results_percent[category] = []
        for val, total in zip(dist_complete, totals):
            if total == 0 and val != 0:
                raise RuntimeError()
            results_percent[category].append(val / total * 100 if total > 0 else 0)

    names = {
        'syntactically correct': 'exact match',
        'exact match': 'exact match',
        'only semantically correct': 'correct',
        'semantically correct': 'correct',
        'incorrect': 'incorrect',
        'invalid': 'invalid',
     }
    # Do the plotting
    # thanks to https://chrisalbon.com/python/data_visualization/matplotlib_percentage_stacked_bar_plot/
    # figure, (hist_ax, dist_ax) = plt.subplots(2, figsize=(12,8))
    figure, (dist_ax) = plt.subplots(1, figsize=(12, 5))
    bar_width = 1
    # hist_ax.bar(x, totals, width=bar_width, color='#3071ff', edgecolor='white')
    # hist_ax.set_ylabel('number of items')
    # hist_ax.set_xlabel('formula size')
    for category, dist_percent in results_percent.items():
        dist_ax.bar(x, dist_percent, bottom=bottom_positions, label=names[category], width=bar_width, color=colors[category], edgecolor='white')
        bottom_positions = [acc + q for acc, q in zip(bottom_positions, dist_percent)]  # update positions
    dist_ax.set_ylabel('percentage')
    dist_ax.set_xlabel('formula size')
    dist_ax.set_ylim(-10, 110)
    dist_ax.legend()
    if 'save_analysis' in kwargs and kwargs['save_analysis'] is not None:
        figure.savefig(kwargs['save_analysis'] + '.png', bbox_inches="tight", dpi=192)
        figure.savefig(kwargs['save_analysis'] + '.svg', bbox_inches="tight", dpi=192)
    # plt.show()

    # collapse size-wise data for further processing
    results_collapsed = {}
    for category, dist in full_results.items():
        results_collapsed[category] = sum(dist.values())
    return results_collapsed


def encode_for_satisfiability(trace_obj: LTLTrace, formula: LTLFormula):
    # prefix
    step_constraints = []
    for idx, trace_step_formula in enumerate(trace_obj.prefix_formulas):
        for _ in range(idx):  # prepend X's for step
            trace_step_formula = F_NEXT(trace_step_formula)
        step_constraints.append(trace_step_formula)
    prefix_part = reduce(F_AND, step_constraints) if step_constraints else None  # AND together

    # generate encoding aps for cycle steps
    cycle_encoding_bits = bin(len(trace_obj.cycle_formulas))[2:]
    used_aps = trace_obj.contained_aps() | formula.contained_aps()  # TODO: remove?
    num_encoding_aps = len(cycle_encoding_bits)
    encoding_aps = ['c' + str(q) for q in range(num_encoding_aps)]

    # build encodings for cycle steps
    encodings = []
    for idx, _ in enumerate(trace_obj.cycle_formulas):
        bin_rep = '{{:0{:d}b}}'.format(num_encoding_aps).format(idx)
        encoding = []
        for idx_encode, c in enumerate(bin_rep):
            ap = F_AP(encoding_aps[idx_encode])
            if c == '1':
                encoding.append(ap)
            elif c == '0':
                encoding.append(F_NOT(ap))
            else:
                raise ValueError()
        encodings.append(reduce(F_AND, encoding))

    # build "chain" between cycle steps
    cycle_constraints = []
    for idx, _ in enumerate(trace_obj.cycle_formulas):
        if idx + 1 == len(trace_obj.cycle_formulas):  # last step in cycle
            next_idx = 0
        else:
            next_idx = idx + 1
        cycle_constraints.append(F_GLOBALLY(F_IMLIES(encodings[idx], F_NEXT(F_AND(encodings[next_idx], trace_obj.cycle_formulas[next_idx])))))
    cycle_part = reduce(F_AND, cycle_constraints)  # and step formulas together, add to complete formula

    # start chain
    cycle_part += F_AND(encodings[0], trace_obj.cycle_formulas[0])

    # prepend nexts to cycle
    for _ in range(len(trace_obj.prefix_formulas)):
        cycle_part = F_NEXT(cycle_part)

    # add Nexts to cycle part, add formula to check
    complete = prefix_part + cycle_part + F_NOT(formula)
    return complete


def process_ltl_item(formula_str, trace_str, target_str, formula_format):
    formula_obj = ltl_parser.ltl_formula(formula_str, format=formula_format)
    target_obj = ltl_parser.ltl_trace(target_str, format=formula_format)
    try:
        trace_obj = ltl_parser.ltl_trace(trace_str, format=formula_format)
    except ltl_parser.ParseError as e:
        return {"result": "invalid", "error": f"{e}"}
    if trace_obj.equal_to(target_obj, extended_eq=True):
        return {"result": "exact match"}
    # spot trace check
    formula_spot = spot.formula(formula_obj.to_str('spot'))
    trace_spot = spot.parse_word(trace_obj.to_str('spot'))
    formula_automaton = formula_spot.translate()
    trace_automaton = trace_spot.as_automaton()
    try:
        spot_holds = spot.contains(formula_automaton, trace_automaton)
        result = "semantically correct" if spot_holds else "incorrect"
        return {"result": result}
    except RuntimeError:
        return {"result": "runtime error"}


def evaluate_ltl(data, polish=True, threads=None, timeout=30):
    """
    Args:
        data: List of tuples (formula, predicted trace, target trace)
    """
    if threads is None:
        threads = multiprocessing.cpu_count()

    results = [{"formula": a, "trace": b, "target": c} for a, b, c in data]

    formula_format = 'network-' + ('polish' if polish else 'infix')
    process_item = partial(process_ltl_item, formula_format=formula_format)

    with multiprocessing.Pool(threads) as pool, tqdm(total=len(data), desc="Evaluate") as pbar:
        def process_handle(handle, i, start_time):
            remaining = timeout - (time.time() - start_time)
            if not handle.ready() and remaining > 0:  # Still alive
                return True
            try:
                result = handle.get(remaining)
            except multiprocessing.TimeoutError:
                result = {"result": "timeout"}
            result.update(results[i])
            result["time"] = time.time() - start_time
            results[i] = result
            pbar.update(1)
            return False
        handles = []
        for index, new_item in enumerate(data):
            while len(handles) == threads:
                # Wait first
                handles = [handle for handle in handles if process_handle(*handle)]
                if len(handles) == threads:
                    # API doesn't allow us to wait on any handle.
                    # Wait on last for a short duration before checking again:
                    handles[-1][0].wait(0.125)
            # Add new task
            future = pool.apply_async(process_item, new_item)
            handles.append((future, index, time.time()))
        # Finish off
        while len(handles) > 0:
            first_len = len(handles)
            handles = [handle for handle in handles if process_handle(*handle)]
            if len(handles) == first_len:
                handles[-1][0].wait(0.125)

    return results


def analyze_results(results):
    """
    Calculate statistics per size from evaluation results.
    """
    output = defaultdict(lambda: defaultdict(int))
    for result in results:
        # NOTE: This assumes that each syntax tree element is single character,
        # which is true for our LTL syntax
        size = len(result["formula"])
        output[result["result"]][size] += 1
    return output
