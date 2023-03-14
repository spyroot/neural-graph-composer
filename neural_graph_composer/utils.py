from typing import Optional

import numpy as np
import torch


def np_onehot(indices, depth, data_type: Optional[bool] = bool):
    """1D array of indices to a one-hot 2D array with given depth
    :param indices:
    :param depth:
    :param data_type:
    :return:
    """
    onehot_seq = np.zeros((len(indices), depth), dtype=data_type)
    onehot_seq[np.arange(len(indices)), indices] = 1.0
    return onehot_seq


def midi_labels():
    """

    """
    labels = [51, 52, 53, 54, 57]
    labels_tensor = torch.LongTensor(labels)
    num_classes = len(set(labels))
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=127)


def pretty_midi_to_one_hot(pm, fs=100):
    """Compute a one hot matrix of a pretty midi object
    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    Returns
    -------
    one_hot : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument. 1 represents Note Ons,
        -1 represents Note offs, 0 represents constant/do-nothing
    """

    # Allocate a matrix of zeros - we will add in as we go
    one_hots = []

    for instrument in pm.instruments:
        one_hot = np.zeros((128, int(fs * instrument.get_end_time()) + 1))
        for note in instrument.notes:
            # note on
            one_hot[note.pitch, int(note.start * fs)] = 1
            print('note on', note.pitch, int(note.start * fs))
            # note off
            one_hot[note.pitch, int(note.end * fs)] = 0
            print('note off', note.pitch, int(note.end * fs))
        one_hots.append(one_hot)

    one_hot = np.zeros((128, np.max([o.shape[1] for o in one_hots])))
    for o in one_hots:
        one_hot[:, :o.shape[1]] += o

    one_hot = np.clip(one_hot, -1, 1)
    return one_hot
