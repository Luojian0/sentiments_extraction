# -*- encoding: utf-8 -*-
'''
@File    :   fastBPE_FE.py
@Time    :   2020/06/26 17:09:26
@Author  :   Luo Jianhui 
@Version :   1.0
@Contact :   kid1412ljh@outlook.com
'''

# here put the import lib
import numpy as np
import pandas as pd
import torch
import argparse

from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# Load BPE encoder
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
                    default="BERTweet_base_transformers/bpe.codes",
                    required=False,
                    type=str,
                    help='path to fastBPE BPE')
args = parser.parse_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("BERTweet_base_transformers/dict.txt")


def encode_label(sent):
    subwords = bpe.encode(sent)
    # Map subword tokens to corresponding indices in the dictionary
    input_ids = vocab.encode_line(subwords,
                                  append_eos=False,
                                  add_if_not_exist=False).long().numpy()
    return input_ids


def label(x, y):
    """Generate the labels of length 48 with padding
    """
    ori = encode_label(x)
    sel = encode_label(y)
    n = len(ori)
    label = np.ones(n, dtype=np.int)
    index_sel = []
    preceding = []
    for i in range(len(sel)):
        if sel[i] in ori:
            if len(preceding) == 0:
                pos = np.argwhere(ori == sel[i])[0][0]
                index_sel.append(pos)
                preceding.append(sel[i])
            else:
                if sel[i] not in preceding:
                    pos = np.argwhere(ori == sel[i])[0][0]
                    index_sel.append(pos)
                    preceding.append(sel[i])
                else:
                    pos = np.argwhere(ori == sel[i]).ravel()[-1]
                    index_sel.append(pos)
    label[index_sel] = 2
    max_len = 63
    label = np.r_[label, np.zeros(max_len - n, dtype=np.int)]
    label = np.r_[0, label, 0]

    return label


def all_label(path):

    clean_data = pd.read_csv(path)
    all_label = []
    for i in tqdm(range(len(clean_data))):
        ori = clean_data.text.values
        sel = clean_data.selected_text.values
        all_label.append(label(ori[i], sel[i]))

    return all_label


def encode(path):

    data = pd.read_csv(path)
    sentences = data.text.values
    labels = all_label(path)

    input_ids_list = []
    attention_masks = []
    max_len = 65

    for sent in tqdm(sentences):

        subwords = '<s> ' + bpe.encode(sent) + ' </s>'
        # Map subword tokens to corresponding indices in the dictionary
        input_ids = vocab.encode_line(subwords,
                                      append_eos=False,
                                      add_if_not_exist=False).long().tolist()
        # Generate attention masks
        if len(input_ids) < max_len:
            non_masks = np.ones(len(input_ids), dtype=np.int).tolist()
            masks = np.zeros(max_len - len(input_ids), dtype=np.int).tolist()
            non_masks.extend(masks)
        else:
            non_masks = np.ones(len(input_ids), dtype=np.int)

        # Add paddings if length less than max_len
        if len(input_ids) < max_len:
            paddings = np.ones(max_len - len(input_ids), dtype=np.int).tolist()
            input_ids.extend(paddings)

        input_ids_list.append(input_ids)
        attention_masks.append(non_masks)

    # Convert the lists into tensors.
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # Save dictionary
    output_vocab_file = "./models/fine_tuning_dict.text"
    vocab.save(output_vocab_file)
    return input_ids, attention_masks, labels

def split_data(path):
    
    input_ids, attention_masks, labels = encode(path)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


if __name__ == '__main__':
    path = 'clean_data.csv'
    train_dataset, val_dataset = split_data(path)