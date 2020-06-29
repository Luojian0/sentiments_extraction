# -*- encoding: utf-8 -*-
'''
@File    :   bertweet_predict.py
@Time    :   2020/06/27 23:12:09
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

from transformers import RobertaForTokenClassification, AdamW, RobertaConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import time
import datetime
import random

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


def encode(path):

    data = pd.read_csv(path)
    sentences = data.text.values

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

    return input_ids, attention_masks


def dataloader(path):

    input_ids, attention_masks = encode(path)
    test_dataset = TensorDataset(input_ids, attention_masks)
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our test sets.
    # For prediction the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
        test_dataset,  # The validation samples.
        sampler=SequentialSampler(
            test_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return test_dataloader


def get_model():

    # Load model
    config = RobertaConfig.from_pretrained(
        "BERTweet_base_transformers/config.json", num_labels=3)
    BERTweet = RobertaForTokenClassification.from_pretrained(
        "BERTweet_base_transformers/model.bin", config=config)

    optimizer = AdamW(
        BERTweet.parameters(),
        lr=1e-05,  # args.learning_rate - default is 5e-5, 
        eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    )

    return BERTweet, optimizer

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def convert_label_to_string(ids, labels):
    input_ids = ids.numpy()
    ow_list = []
    for i in labels.shape[0]: 
        ind = np.argwhere(labels[i]== 2).ravel()
        sel_ids = input_ids[i][ind]
        subwords = vocab.string(sel_ids, bpe_symbol = '')
        origin_words = bpe.decode(subwords)
        ow_list.append(origin_words)
    return ow_list


def predict(path):


    test_dataloader = dataloader(path)
    model, _ = get_model()

    # Tell pytorch to run this model on the GPU.
    # model.cuda()
    model.cpu()
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Prediction...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # The list save all predictions
    preds = []

    # Evaluate data for one epoch
    for batch in test_dataloader:

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask)


        # Calculate the labels for this batch of test sentences, and save it
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=-1)
        origin_str = convert_label_to_string(b_input_ids, pred_flat)
        preds.extend(origin_str)

    # Measure how long the validation run took.
    prediction_time = format_time(time.time() - t0)

    print("  Prediction took: {:}".format(prediction_time))

    print("")
    print("Prediction complete!")


