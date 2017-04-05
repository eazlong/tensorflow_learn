
"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import jieba

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model as rnn_seq2seq

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("cn_vocab_size", 40000, "Chinese vocabulary size.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model/", "Train directory")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (35, 40), (50, 60)]


def read_data(source_path, target_path):
    data_set = [[] for _ in _buckets]

    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 10000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()

                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(sess, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = rnn_seq2seq.Seq2SeqModel(
        FLAGS.cn_vocab_size, FLAGS.en_vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.initialize_all_variables())
    return model


def train():
    # Prepare data.
    print("Preparing data in %s" % FLAGS.data_dir)
    source_path = os.path.join(FLAGS.data_dir, "source_token_ids.txt")
    target_path = os.path.join(FLAGS.data_dir, "target_token_ids.txt")

    with tf.Session() as sess:
        # Create model.
        model = create_model(sess, False)
        train_set = read_data(source_path, target_path)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        current_step = 0
        previous_losses = []
        while True:
            step_time, loss = 0.0, 0.0
            for _ in range(FLAGS.steps_per_checkpoint):
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])

                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics, and run evals.

            # Print statistics for the previous epoch.
            # perplexity = math.exp(loss) if loss < 300 else float('inf')
            print("global step %d learning rate %.4f step-time %.2f loss "
                  "%.6f" % (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, loss))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # load vocab
        cn_vocab_path = os.path.join(FLAGS.data_dir, "source_vocab.txt")
        en_vocab_path = os.path.join(FLAGS.data_dir, "target_vocab.txt")

        cn_vocab, _ = data_utils.initialize_vocabulary(cn_vocab_path)
        _, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            seg_list = jieba.lcut(sentence)
            # print(" ".join(seg_list))
            token_ids = [cn_vocab.get(w.encode(encoding="utf-8"), data_utils.UNK_ID) for w in seg_list]
            # token_ids = data_utils.sentence_to_token_ids(sentence, cn_vocab)
            bucket_id = min([b for b in range(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            output = " ".join([tf.compat.as_str(rev_en_vocab[output]) for output in outputs])
            # print(" ".join([tf.compat.as_str(rev_en_vocab[output]) for output in outputs]))
            print(output.capitalize())
            print("\n")
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):
    # python translate.py --decode True
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
