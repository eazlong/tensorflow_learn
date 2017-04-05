
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

source_max_size = 20
target_max_size = 26

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def maybe_load(directory, filename):
    """Load filename from directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)

    filepath = os.path.join(directory, filename)
    return filepath


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = sentence.strip().split()
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None):
    """Create vocabulary file from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to create vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            for line in f:
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)

                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.
    Args:
        vocabulary_path: path to the file containing the vocabulary.
    Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    """
    Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
        sentence: the sentence in bytes format to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.

    Returns:
        a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None):
    """
    Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _= initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                for line in data_file:
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


# def read_data(source_ids_path, target_ids_path):
#     data_set = []
#
#     with tf.gfile.GFile(source_ids_path, mode="r") as source_file:
#         with tf.gfile.GFile(target_ids_path, mode="r") as target_file:
#             source, target = source_file.readline(), target_file.readline()
#
#             while source and target:
#                 source_ids = [int(x) for x in source.split()]
#                 target_ids = [int(x) for x in target.split()]
#                 y = [int(x) for x in target.split()]
#                 # target_ids.append(EOS_ID)
#                 y.append(EOS_ID)
#
#                 data_set.append([source_ids, target_ids, y])
#                 source, target = source_file.readline(), target_file.readline()
#     return data_set


def get_batch(data_set, batch_size):
    encoder_size, decoder_size = source_max_size, target_max_size
    encoder_inputs, decoder_inputs, y_inputs = [], [], []

    for _ in range(batch_size):
        encoder_input, decoder_input, y = random.choice(data_set)

        # Encoder inputs are padded and then reversed.
        encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(encoder_input + encoder_pad))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([GO_ID] + decoder_input +
                              [PAD_ID] * decoder_pad_size)

        y_pad_size = decoder_size - len(y)
        y_inputs.append(y + [PAD_ID] * y_pad_size)

    # Now we create batch-major vectors from the data selected above.
    # batch_encoder_inputs, batch_decoder_inputs = [], []
    batch_encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    # batch_seq_length = np.array(seq_length, dtype=np.int32)
    batch_decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    batch_y_inputs = np.array(y_inputs, dtype=np.int32)

    # # Batch encoder inputs are just re-indexed encoder_inputs.
    # for length_idx in xrange(encoder_size):
    #     batch_encoder_inputs.append(
    #         np.array([encoder_inputs[batch_idx][length_idx]
    #                   for batch_idx in xrange(batch_size)], dtype=np.int32))
    #
    # # Batch decoder inputs are re-indexed decoder_inputs.
    # for length_idx in xrange(decoder_size):
    #     batch_decoder_inputs.append(
    #         np.array([decoder_inputs[batch_idx][length_idx]
    #                   for batch_idx in xrange(batch_size)], dtype=np.int32))

    return batch_encoder_inputs, batch_decoder_inputs, batch_y_inputs


def prepare_data(data_dir):
    source_path = maybe_load(data_dir, "Source_CHN.ch")
    target_path = maybe_load(data_dir, "Target_ENG.en")

    create_vocabulary("data/source_vocab.txt", source_path, 40000)
    create_vocabulary("data/target_vocab.txt", target_path, 40000)

    # source_vocab, source_rev_vocab = initialize_vocabulary("data/source_vocab.txt")
    data_to_token_ids(source_path, "data/source_token_ids.txt", "data/source_vocab.txt")
    data_to_token_ids(target_path, "data/target_token_ids.txt", "data/target_vocab.txt")

    """
    source_data = open(source_path, "r")
    source_words = []
    for line in source_data:
        line = unicode(line, "gbk")
        source_words.append(basic_tokenizer(line))

    print(source_words)
    """

if __name__ == "__main__":
    prepare_data("data/")
