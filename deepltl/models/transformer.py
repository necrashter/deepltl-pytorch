""" Transformer implementation based on https://github.com/tensorflow/models/tree/master/official/nlp/transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from deepltl.layers import attention
from deepltl.layers import positional_encoding as pe
from deepltl.models.beam_search import BeamSearch


def get_activation(activation):
    """
    Args:
        activation: str, name of the activation function
    """
    if activation =='relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation function {activation}')


class AccuracyTracker:
    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                dtype: torch.dtype, datatype for floating point computations
                target_pad_id: int, encodes the padding token for targets
        """
        self.target_pad_id = params['target_pad_id']
        # Ignore dtype for now
        self.reset()
    
    def reset(self):
        # Accuracy
        self.acc = 0.0
        self.acc_total = 0.0
        # Accuracy per sequence
        self.aps = 0.0
        self.aps_total = 0.0

    def record(self, predictions, targets):
        # For ignoring pad tokens
        weights = (targets != self.target_pad_id)
        outputs = torch.argmax(predictions, dim=-1)
        targets = targets.type(torch.int32)

        # Accuracy
        correct_predictions = (outputs == targets) * weights
        self.acc += correct_predictions.sum().item()
        self.acc_total += weights.sum().item()

        # Accuracy per sequence
        incorrect_predictions = (outputs != targets) * weights
        correct_sequences = 1.0 - torch.minimum(torch.tensor(1.0), incorrect_predictions.sum(dim=-1))
        self.aps += correct_sequences.sum().item()
        self.aps_total += torch.numel(correct_sequences)

    def write(self, writer: SummaryWriter, step, kind: str):
        """
        Write and reset the accuracy tracker.
        Args:
            writer: tensorboard SummaryWriter object
            step: int, current training step
            kind: str, either 'train' or 'val'
        """
        if self.acc_total == 0 or self.aps_total == 0:
            raise ValueError('No data to write')
        writer.add_scalar(f'accuracy/{kind}', self.acc / self.acc_total, step)
        writer.add_scalar(f'accuracy_per_sequence/{kind}', self.aps / self.aps_total, step)
        self.reset()



class TransformerEncoderLayer(nn.Module):
    """A single encoder layer of the Transformer that consists of two sub-layers: a multi-head
    self-attention mechanism followed by a fully-connected feed-forward network. Both sub-layers
    employ a residual connection followed by a layer normalization."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        self.params = params

        self.multi_head_attn = attention.MultiHeadAttention(
            params['d_embed_enc'], params['num_heads'])

        self.ff = nn.Sequential(
            nn.Linear(params['d_embed_enc'], params['d_ff']),
            get_activation(params['ff_activation']),
            nn.Linear(params['d_ff'], params['d_embed_enc'])
        )

        self.norm_attn = nn.LayerNorm(params['d_embed_enc'])
        self.norm_ff = nn.LayerNorm(params['d_embed_enc'])

        self.dropout_attn = nn.Dropout(params['dropout'])
        self.dropout_ff = nn.Dropout(params['dropout'])

    def forward(self, input, mask):
        """
        Args:
            input: float tensor with shape (batch_size, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, input_length)
        """
        attn, attn_weights = self.multi_head_attn(input, input, input, mask)
        attn = self.dropout_attn(attn)
        norm_attn = self.norm_attn(attn + input)

        ff_out = self.ff(norm_attn)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + norm_attn)

        return norm_ff_out, attn_weights


class TransformerDecoderLayer(nn.Module):
    """A single decoder layer of the Transformer that consists of three sub-layers: a multi-head
    self-attention mechanism followed by a multi-head encoder-decoder-attention mechanism followed
    by a fully-connected feed-forward network. All three sub-layers employ a residual connection
    followed by a layer normalization."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerDecoderLayer, self).__init__()
        self.params = params

        self.multi_head_self_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'])
        self.multi_head_enc_dec_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'])

        self.ff = nn.Sequential(
            nn.Linear(params['d_embed_dec'], params['d_ff']),
            get_activation(params['ff_activation']),
            nn.Linear(params['d_ff'], params['d_embed_dec'])
        )

        self.norm_self_attn = nn.LayerNorm(params['d_embed_dec'])
        self.norm_enc_dec_attn = nn.LayerNorm(params['d_embed_dec'])
        self.norm_ff = nn.LayerNorm(params['d_embed_dec'])

        self.dropout_self_attn = nn.Dropout(params['dropout'])
        self.dropout_enc_dec_attn = nn.Dropout(params['dropout'])
        self.dropout_ff = nn.Dropout(params['dropout'])

    def forward(self, input, enc_output, look_ahead_mask, padding_mask, cache=None):
        """
        Args:
            input: float tensor with shape (batch_size, target_length, d_embed_dec)
            enc_output: float tensor with shape (batch_size, input_length, d_embed_enc)
            look_ahead_mask: float tensor with shape (1, 1, target_length, target_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            cache: dict
        """
        self_attn, self_attn_weights = self.multi_head_self_attn(
            input, input, input, look_ahead_mask, cache)
        self_attn = self.dropout_self_attn(self_attn)
        norm_self_attn = self.norm_self_attn(self_attn + input)

        enc_dec_attn, enc_dec_attn_weights = self.multi_head_enc_dec_attn(
            norm_self_attn, enc_output, enc_output, padding_mask)
        enc_dec_attn = self.dropout_enc_dec_attn(enc_dec_attn)
        norm_enc_dec_attn = self.norm_enc_dec_attn(
            enc_dec_attn + norm_self_attn)

        ff_out = self.ff(norm_enc_dec_attn)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + norm_enc_dec_attn)

        return norm_ff_out, self_attn_weights, enc_dec_attn_weights


class TransformerEncoder(nn.Module):
    """The encoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                input_vocab_size: int, size of input vocabulary
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.params = params
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(params) for _ in range(params['num_layers'])])

    def forward(self, x, padding_mask):
        attn_weights = {}
        for i, layer in enumerate(self.enc_layers):
            x, self_attn_weights = layer(x, padding_mask)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
        return x, attn_weights


class TransformerDecoder(nn.Module):
    """The decoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
                target_vocab_size: int, size of target vocabulary         
        """
        super(TransformerDecoder, self).__init__()
        self.params = params
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(params) for _ in range(params['num_layers'])])

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, cache=None):
        attn_weights = {}
        for i, layer in enumerate(self.dec_layers):
            layer_cache = cache[f'layer_{i}'] if cache is not None else None
            x, self_attn_weights, enc_dec_attn_weights = layer(x, enc_output, look_ahead_mask, padding_mask, layer_cache)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
            attn_weights[f'layer_{i+1}']['enc_dec_attn'] = enc_dec_attn_weights
        return x, attn_weights


class Transformer(nn.Module):
    """The Transformer that consists of an encoder and a decoder. The encoder maps the input
    sequence to a sequence of continuous representations. The decoder then generates an output
    sequence in an auto - regressive way."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                alpha: float, strength of normalization in beam search algorithm
                beam_size: int, number of beams kept by beam search algorithm
                d_embed_enc: int, dimension of encoder embedding
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layer
                input_vocab_size: int, size of input vocabulary
                max_encode_length: int, maximum length of input sequence
                max_decode_length: int, maximum lenght of target sequence
                dropout: float, percentage of droped out units
                dtype: datatype for floating point computations
                target_start_id: int, encodes the start token for targets
                target_eos_id: int, encodes the end of string token for targets
                target_vocab_size: int, size of target vocabulary
        """
        super(Transformer, self).__init__()
        self.params = params

        self.encoder_embedding = nn.Embedding(params['input_vocab_size'], params['d_embed_enc'])
        self.register_buffer(
            'encoder_positional_encoding',
            pe.positional_encoding(params['max_encode_length'], params['d_embed_enc']),
            persistent=False,
        )
        self.encoder_dropout = nn.Dropout(params['dropout'])

        self.encoder_stack = TransformerEncoder(params)

        self.decoder_embedding = nn.Embedding(params['target_vocab_size'], params['d_embed_dec'])
        self.register_buffer(
            'decoder_positional_encoding',
            pe.positional_encoding(params['max_decode_length'], params['d_embed_dec']),
            persistent=False,
        )
        self.decoder_dropout = nn.Dropout(params['dropout'])

        self.decoder_stack = TransformerDecoder(params)

        self.final_projection = nn.Linear(params['d_embed_dec'], params['target_vocab_size'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, target=None, positional_encoding=None):
        """
        Args:
            input: int tensor with shape (batch_size, input_length)
            (optional) target: int tensor with shape (batch_size, target_length)
            (optional) positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        input_padding_mask = create_padding_mask(input, self.params['input_pad_id'], self.params['dtype'])

        if positional_encoding is None:
            seq_len = input.size(1)
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding)

        if target is not None:
            probs, _ = self.decode(target, encoder_output, input_padding_mask)
            return probs
        else:
            return self.predict(encoder_output, encoder_attn_weights, input_padding_mask)

    def encode(self, inputs, padding_mask, positional_encoding):
        """
        Args:
            inputs: int tensor with shape (batch_size, input_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc)
        """
        input_embedding = self.encoder_embedding(inputs)
        input_embedding *= torch.sqrt(torch.tensor(self.params['d_embed_enc'], dtype=self.params['dtype']))
        input_embedding += positional_encoding
        input_embedding = self.encoder_dropout(input_embedding)
        encoder_output, attn_weights = self.encoder_stack(input_embedding, padding_mask)
        return encoder_output, attn_weights

    def decode(self, target, encoder_output, input_padding_mask):
        """
        Args:
            target: int tensor with shape (batch_size, target_length)
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
        
        Returns:
            logits: float tensor with shape (batch_size, target_length, target_vocab_size)
            attn_weights: dictionary with keys 'layer_i' where i is the layer number and values are float tensors with shape (batch_size, num_heads, target_length, input_length)
        """
        target_length = target.size(1)
        look_ahead_mask = create_look_ahead_mask(target_length, target.device, self.params['dtype'])
        target_padding_mask = create_padding_mask(target, self.params['input_pad_id'], self.params['dtype'])
        look_ahead_mask = torch.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = F.pad(target, (1, 0), value=self.params['target_start_id'])[:, :-1]

        target_embedding = self.decoder_embedding(target)  # (batch_size, target_length, d_embedding)
        target_embedding *= torch.sqrt(torch.tensor(self.params['d_embed_dec'], dtype=torch.float32))
        target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding)

        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding, encoder_output, look_ahead_mask, input_padding_mask)
        output = self.final_projection(decoder_output)
        # Note that PyTorch's CrossEntropy Loss applies softmax
        # We will output logits
        # probs = self.softmax(output)

        return output, attn_weights

    def predict(self, encoder_output, encoder_attn_weights, input_padding_mask):
        """
        Args:
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            encoder_attn_weights: dictionary, self attention weights of the encoder
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
        """
        batch_size = encoder_output.size(0)

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            decoder_input *= torch.sqrt(torch.tensor(self.params['d_embed_dec'], dtype=self.params['dtype']))
            decoder_input += self.decoder_positional_encoding[:, i:i + 1, :]

            look_ahead_mask = create_look_ahead_mask(self.params['max_decode_length'], ids.device, self.params['dtype'])
            self_attention_mask = look_ahead_mask[:, :, i:i + 1, :i + 1]
            decoder_output, _ = self.decoder_stack(
                decoder_input, cache['encoder_output'], self_attention_mask, cache['input_padding_mask'], cache)
            output = self.final_projection(decoder_output)
            probs = self.softmax(output)
            probs = probs.squeeze(1)
            return probs, cache

        initial_ids = torch.ones(batch_size, dtype=torch.int32, device=encoder_output.device) * self.params['target_start_id']

        num_heads = self.params['num_heads']
        d_heads = self.params['d_embed_dec'] // num_heads
        # create cache structure for decoder attention
        cache = {
            f'layer_{layer}': {
                'keys': torch.zeros(batch_size, 0, num_heads, d_heads, device=encoder_output.device, dtype=self.params['dtype']),
                'values': torch.zeros(batch_size, 0, num_heads, d_heads, device=encoder_output.device, dtype=self.params['dtype'])
            } for layer in range(self.params['num_layers'])
        }
        # add encoder output to cache
        cache['encoder_output'] = encoder_output
        cache['input_padding_mask'] = input_padding_mask

        beam_search = BeamSearch(logits_fn, batch_size, encoder_output.device, self.params)
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        # compute attention weights
        _, decoder_attn_weights = self.decode(top_decoded_ids, encoder_output, input_padding_mask)

        return {'outputs': top_decoded_ids, 'scores': top_scores, 'enc_attn_weights': encoder_attn_weights, 'dec_attn_weights': decoder_attn_weights}


def create_padding_mask(input, pad_id, dtype=torch.float32):
    """
    Args:
        input: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: data type of look ahead mask
    Returns:
        padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = (input == pad_id).to(dtype)
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size, device, dtype=torch.float32):
    """
    creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        device: torch.device, device where the tensors reside
        dtype: data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1)