# pylint: disable = line-too-long
import os
import os.path as path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import urllib.request
from tqdm import tqdm

from deepltl.train.common import argparser, setup, log_params, get_latest_checkpoint, CustomPadCollate, get_run_path
from deepltl.train.common import test_and_analyze_ltl, test_and_analyze_sat
from deepltl.optimization.lr_schedules import TransformerSchedule
from deepltl.models.transformer import AccuracyTracker, Transformer
from deepltl.data import vocabulary, datasets
from deepltl.data.datasets import get_dataset_splits

def download_dataset(dataset_name, problem, splits, data_dir):
    if not path.isdir(data_dir):
        os.mkdir(data_dir)

    url = 'https://storage.googleapis.com/deepltl_data/data/'

    if problem == 'ltl':
        url += 'ltl_traces/'
        dataset_dir = path.join(data_dir, 'ltl')
    if problem == 'prop':
        url += 'sat/'
        dataset_dir = path.join(data_dir, 'prop')
    
    if not path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    url += dataset_name + '/'

    dataset_dir = path.join(dataset_dir, dataset_name)
    if not path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    # Check if splits already exist
    for split in splits:
        split_dir = path.join(dataset_dir, split + '.txt')
        if not path.isfile(split_dir):
            print(f'Downloading {split} ...')
            urllib.request.urlretrieve(url + split + '.txt', split_dir)


def run():
    # Argument parsing
    parser = argparser()
    # add specific arguments
    parser.add_argument('--problem', type=str, default='ltl', help='available problems: ltl, prop')
    parser.add_argument('--d-embed-enc', type=int, default=128)
    parser.add_argument('--d-embed-dec', type=int, default=None)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--ff-activation', default='relu')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--tree-pos-enc', action='store_true', default=False, help='use tree positional encoding')
    parser.add_argument('--layer-norm-eps', type=float, default=1e-6, help='Epsilon value used in layer norm')
    parser.add_argument('--seed', type=int, help='Seed for the random number generator')
    params = parser.parse_args()
    setup(**vars(params))

    aps = ['a', 'b', 'c', 'd', 'e']
    consts = ['0', '1']

    if params.problem != 'ltl' and params.problem != 'prop':
        sys.exit(f'{params.problem} is not a valid problem\n')

    # Dataset specification
    if params.ds_name is None:
        sys.exit('No dataset specified\n')
    else:
        if params.ds_name == 'ltl-35' or params.ds_name == 'prop-35':
            params.max_encode_length = 37
            dataset_name = 'na-5-ts-35-nf-1m-lbt-sat'
        elif params.ds_name == 'ltl-50-test' or params.ds_name == 'prop-50-test':
            params.max_encode_length = 52
            if not params.test: # train mode
                sys.exit(f'Dataset {params.ds_name} can only be used in test mode\n')
            dataset_name = 'na-5-ts-35-50-nf-20k-lbt-sat'
        elif params.ds_name == 'prop-60-no-derived':
            params.max_encode_length = 62
            aps += ['f', 'g', 'h', 'i', 'j']
            dataset_name = 'lessops_na-10-ts-60-nf-1m-lbt-sat'
        else:
            sys.exit(f'{params.ds_name} is not a valid dataset\n')

    if not params.test: # train mode
        download_dataset(dataset_name, params.problem, ['train', 'val', 'test'], params.data_dir)
    else: # test only
        download_dataset(dataset_name, params.problem, ['test'], params.data_dir)
    
    if params.problem == 'ltl':
        input_vocab = vocabulary.LTLVocabulary(aps=aps, consts=consts, ops=['U', 'X', '!', '&'], eos=not params.tree_pos_enc)
        target_vocab = vocabulary.TraceVocabulary(aps=aps, consts=consts, ops=['&', '|', '!'])
        params.max_decode_length = 64
        data_dir = path.join(params.data_dir, 'ltl')
        dataset_class = datasets.LTLTracesDataset
        dataset_args = {
            'ltl_vocab': input_vocab,
            'trace_vocab': target_vocab,
            'max_length_formula': params.max_encode_length - 2,
            'max_length_trace': params.max_decode_length - 2,
            'prepend_start_token': False,
            'tree_pos_enc': params.tree_pos_enc,
        }
    elif params.problem == 'prop':
        input_vocab = vocabulary.LTLVocabulary(aps, consts, ['!', '&', '|', '<->', 'xor'], eos=not params.tree_pos_enc)
        target_vocab = vocabulary.TraceVocabulary(aps, consts, [], special=[])
        data_dir = path.join(params.data_dir, 'prop')
        dataset_class = datasets.BooleanSatDataset
        dataset_args = {
            'formula_vocab': input_vocab,
            'assignment_vocab': target_vocab,
            'tree_pos_enc': params.tree_pos_enc,
        }
        if params.ds_name == 'prop-60-no-derived':
            params.max_decode_length = 22
        else:
            params.max_decode_length = 12

    params.input_vocab_size = input_vocab.vocab_size()
    params.input_pad_id = input_vocab.pad_id
    params.target_vocab_size = target_vocab.vocab_size()
    params.target_start_id = target_vocab.start_id
    params.target_eos_id = target_vocab.eos_id
    params.target_pad_id = target_vocab.pad_id
    params.dtype = torch.float32

    if params.d_embed_dec is None:
        params.d_embed_dec = params.d_embed_enc
    print('Specified dimension of encoder embedding:', params.d_embed_enc)
    params.d_embed_enc -= params.d_embed_enc % params.num_heads  # round down
    print('Specified dimension of decoder embedding:', params.d_embed_dec)
    params.d_embed_dec -= params.d_embed_dec % params.num_heads  # round down
    print('Parameters:')
    for key, val in vars(params).items():
        print('{:25} : {}'.format(key, val))

    device = torch.device(params.device)
    print("Using device:", device)

    collate_fn = CustomPadCollate(params.d_embed_enc if params.tree_pos_enc else None)
    if not params.test: # train mode
        train_dataset, val_dataset, test_dataset = get_dataset_splits(dataset_name, ['train', 'val', 'test'], dataset_class, dataset_args, data_dir)
        # NOTE: Tensorflow implementation used to drop the last batch (drop_last=True in PyTorch Dataloader)
        train_dataloader = data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    else:  # test mode
        test_dataset, = get_dataset_splits(dataset_name, ['test'], dataset_class, dataset_args, data_dir)

    checkpoint_path = get_run_path("checkpoints", **vars(params))
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)

    if not params.test:  # train mode
        # Model & Training specification
        model = Transformer(vars(params)).to(device)
        # lr is 1 so that it's determined solely by the scheduler
        optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id, reduction="sum")
        lr_scheduler = TransformerSchedule(optimizer, params.d_embed_enc, warmup_steps=params.warmup_steps)
        if latest_checkpoint:
            model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
            print(f'Loaded weights from checkpoint {latest_checkpoint}')
        else:
            print('No checkpoint found, training from scratch')

        # TODO: Add early stopping
        # callbacks = [tf.keras.callbacks.EarlyStopping('val_accuracy', patience=4, restore_best_weights=True)]

        writer = SummaryWriter(log_dir=get_run_path("tensorboard", **vars(params)))
        acc_tracker = AccuracyTracker(vars(params))
        log_params(**vars(params))

        # Train!
        train_iter = 0
        print("Training begins")
        for epoch in range(params.epochs):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, leave=False, desc=f'Train {epoch + 1}/{params.epochs}'):
                batch = tuple(x.to(device) for x in batch)
                # batch = (input, target) or (input, target, positional_encoding)
                optimizer.zero_grad()
                outputs = model(*batch)  # outputs have the size (batch, seq_len, vocab_size)
                acc_tracker.record(outputs, batch[1])
                loss = criterion(outputs.flatten(end_dim=-2), batch[1].flatten())

                train_loss += loss.item()
                writer.add_scalar("batch/loss", loss.item(), train_iter)
                writer.add_scalar("batch/lr", lr_scheduler.get_last_lr()[0], train_iter)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_iter += 1
            train_loss /= len(train_dataset)
            acc_tracker.write(writer, epoch, "train")

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_batch in tqdm(val_dataloader, leave=False, desc=f'Val {epoch + 1}/{params.epochs}'):
                    val_batch = tuple(x.to(device) for x in val_batch)
                    val_outputs = model(*val_batch)
                    acc_tracker.record(val_outputs, val_batch[1])
                    val_loss += criterion(val_outputs.flatten(end_dim=-2), val_batch[1].flatten()).item()
                val_loss /= len(val_dataset)
                acc_tracker.write(writer, epoch, "val")

            print(f'Epoch {epoch+1}/{params.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.flush()

            # Save model
            os.makedirs(checkpoint_path, exist_ok=True)
            filepath = os.path.join(checkpoint_path, f'ep{epoch+1:03d}_vl{val_loss:.3f}.pth')
            torch.save(model.state_dict(), filepath)
    
        writer.close()
    else:  # test mode
        prediction_model = Transformer(vars(params)).to(device)
        if latest_checkpoint:
            prediction_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
            print(f'Loaded weights from checkpoint {latest_checkpoint}')
        else:
            sys.exit('Could not load weights from checkpoint')
        sys.stdout.flush()
        prediction_model.eval()

        test_subset = data.Subset(test_dataset, torch.arange(100))
        test_dataloader = data.DataLoader(test_subset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

        if params.problem == 'ltl':
            if params.tree_pos_enc:
                def pred_fn(x, pe):
                    # Target is still none
                    output = prediction_model(x, None, pe)
                    return output["outputs"]
            else:
                def pred_fn(x):
                    output = prediction_model(x)
                    return output["outputs"]
            test_and_analyze_ltl(pred_fn, test_dataloader, device, input_vocab, target_vocab, log_name='test.log', **vars(params))
        elif params.problem == 'prop':
            test_and_analyze_sat(prediction_model, test_dataloader, device, input_vocab, target_vocab, log_name='test.log', **vars(params))
        else:
            raise ValueError(f'Unknown problem {params.problem}')


if __name__ == '__main__':
    run()
