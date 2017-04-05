import numpy as np
import data_prepare as dp
import tensorflow as tf
import collections

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
weights = tf.Variable( tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))

# Embeddings for examples: [batch_size, embedding_size]
example_emb = tf.nn.embedding_lookup(embeddings, examples)
# Weights for labels: [batch_size, embedding_size]
true_w = tf.nn.embedding_lookup(weights, labels)
# Biases for labels: [batch_size, 1]
true_b = tf.nn.embedding_lookup(biases, labels)

labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [batch_size, 1])

# Negative sampling.
sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
     true_classes=labels_matrix,
     num_true=1,
     num_sampled=num_samples,
     unique=True,
     range_max=vocab_size,
     distortion=0.75,
     unigrams=vocab_counts.tolist())

# Weights for sampled ids: [num_sampled, embedding_size]
sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
# Biases for sampled ids: [num_sampled, 1]
sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)

# True logits: [batch_size, 1]
true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b
# Sampled logits: [batch_size, num_sampled]
# We replicate sampled noise lables for all examples in the batch
# using the matmul.
sampled_b_vec = tf.reshape(sampled_b, [num_samples])
sampled_logits = tf.matmul(example_emb,
                           sampled_w,
                           transpose_b=True) + sampled_b_vec
# cross-entropy(logits, labels)
true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
   true_logits, tf.ones_like(true_logits))
sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
   sampled_logits, tf.zeros_like(sampled_logits))
# NCE-loss is the sum of the true and noise (sampled words)
# contributions, averaged over the batch.
loss = (tf.reduce_sum(true_xent) +
        tf.reduce_sum(sampled_xent)) / batch_size

lr = init_learning_rate * tf.maximum(
     0.0001, 1.0 - tf.cast(trained_words, tf.float32) / words_to_train)

optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss,
                           global_step=global_step,
                           gate_gradients=optimizer.GATE_NONE)
session.run(train)


lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

initial_state = cell.zero_state(batch_size, tf.float32)
embedding = tf.get_variable("embedding", [vocab_size, size])
# input_data: [batch_size, num_steps]
# targets： [batch_size, num_steps]
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int32, [batch_size, num_steps])
inputs = tf.nn.embedding_lookup(embedding, input_data)
outputs = []
for time_step in range(num_steps):
  (cell_output, state) = cell(inputs[:, time_step, :], state)
  outputs.append(cell_output)

output = tf.reshape(tf.concat(1, outputs), [-1, size])
softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
softmax_b = tf.get_variable("softmax_b", [vocab_size])
logits = tf.matmul(output, softmax_w) + softmax_b

loss = tf.nn.seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(targets, [-1])],
    [tf.ones([batch_size * num_steps])])

optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)
for i in range(max_epoch):
  _, final_state = session.run([train_op, state],
                               {input_data: x,
                                targets: y})

(cell_output, _) = cell(inputs, state)
session.run(cell_output)


