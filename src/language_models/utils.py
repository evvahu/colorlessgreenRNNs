# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import numpy as np


def repackage_hidden(h):
    """Detaches hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def ids_to_embs(data, idtow, ft_model, embsize=300):
    words = torch.empty(size=(data.size(0), data.size(1), embsize))
    #words = np.empty((data.size(0), data.size(1)), dtype=float)
    #data_flat = data.view(-1, 1)
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            #row = data.select(0, i)
            #id = row[j]
            word = idtow[data[i][j].item()]
            words[i, j] = torch.from_numpy(ft_model.get_word_vector(word))
    return words
