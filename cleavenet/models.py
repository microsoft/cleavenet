import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import L2

import cleavenet.data
from cleavenet import analysis, plotter
from cleavenet.utils import mmps 


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom schedule from original Transformer paper https://arxiv.org/abs/1706.03762
    Adapted from https://www.tensorflow.org/text/tutorials/transformer#set_up_the_optimizer
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        #print(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def positional_encoding(length, depth):
    "positional encoding for transformer, from https://www.tensorflow.org/text/tutorials/transformer"
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Looks up a tokens embedding vector, and adds a positional encoding vector

    label=True when input has a conditional tag (z-scores)
    """
    def __init__(self, vocab_size, d_model, max_seq_length=15, mask_zero=True, label=False, start_idx=20):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=mask_zero)
        self.pos_encoding = positional_encoding(length=max_seq_length, depth=d_model)
        self.label = label
        self.start_idx = start_idx
        if label: 
            self.label_embedding = tf.keras.layers.Dense(d_model)

    def call(self, x):
        if self.label:
            x, label = x 
            label_emb = []
            #print("label", label)
            for i in range(len(label)):
                if label[i][0] == self.start_idx: # if start token 
                    label_emb_temp = self.embedding(label[i])
                else:
                    label_emb_temp = self.label_embedding(tf.expand_dims(label[i], 0))
                #print("embedded", label_emb_temp)
                label_emb.append(label_emb_temp)
            label_emb = tf.stack(label_emb)
        x = self.embedding(x)
        if self.label:
            if x.shape[1] == 0:
                x = label_emb
            else:
                x = tf.concat([label_emb, x], 1)
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


### Transformer Components ###
class BaseAttention(tf.keras.layers.Layer):
    """
    Base attention layer, contains component layers;
    - contains a multihead attention, and layers.Add + layers.LayersNormalization
    e.g. Each Nx of a Transformer model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    """
    Global self attention layer; processing context sequence, and propegating information along length
    (information flows in both directions)

    x: target sequence (query, and in this case also the value)
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """
    Causal self attention; similar to global self attention but for output sequence
    To build a casual self attention layer, use appropriate mask when computing attention scores and summing attention
    value
    """
    def call(self, x):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            return_attention_scores=True) # causal self attention layer; each location only has access to locations before it
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    """
    Feed forward network; consists of 2 linear layers, with RELU activation and dropout layer
    x: embedded input B, T, dim
    """
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation=activation),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer; Global self attention and feed forward
    """
    def __init__(self,*, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff, activation=activation, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Encoder; Positional embedding, stack of N encoder layers
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, activation='relu', dropout_rate=0.1, mask_zero=False):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model, mask_zero=mask_zero)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         activation=activation,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        x = self.dropout(x)
        for i in range(self.num_layers):
          x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    "Decoder layer; Causal self attention, feed forward layer "
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.causal_self_attention(x=x)
        self.last_attn_scores = self.causal_self_attention.last_attn_scores
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder; positional embedding, stack of decoder layers
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)
        for i in range(self.num_layers):
          x  = self.dec_layers[i](x)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x #  (batch_size, target_seq_len, d_model).


### Transformer Models ###
class TransformerDecoder(tf.keras.Model):
    "Transformer; Decoder-only model for autoregressive modeling"
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.vocab_size=vocab_size

    def call(self, x):
        x = self.decoder(x) # (batch_size, target_len, d_model)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits

    def compute_loss(self, y, y_hat):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, y, y_hat):
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y, y_hat)
        return m.result()


class ConditionalTransformerDecoder(tf.keras.Model):
    "Transformer; Decoder-only model for autoregressive modeling"
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, label=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None
        self.vocab_size=vocab_size
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    # def build(self, input_shape): # Don't use with new TF
    #     self.pos_embedding.build(input_shape)
    #     input_shape = self.pos_embedding.compute_output_shape(input_shape)
    #     self.dec_layers.build(input_shape)
    #     input_shape = self.dec_layers.compute_output_shape(input_shape)
    #     self.final_layer.build(input_shape)
    #     self.built = True

    def call(self, x):
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)
        for i in range(self.num_layers):
          x  = self.dec_layers[i](x)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits

    def compute_loss(self, y, y_hat):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, y, y_hat):
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y, y_hat)
        return m.result()
        

class TransformerEncoder(tf.keras.Model):
    "Transformer; Encoder-only model for BERT MLM"
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1, output_dim=None, pool_outputs=False, mask_zero=False):
        super().__init__()
        if output_dim == None:
            output_dim=vocab_size
        self.pool_outputs = pool_outputs
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate, mask_zero=mask_zero)
        self.final_layer = tf.keras.layers.Dense(output_dim)
        self.vocab_size = vocab_size

    def call(self, x):
        x = self.encoder(x)  # (batch_size, target_len, d_model)
        self.last_layer_embeddings = x
        if self.pool_outputs:
            x = tf.squeeze(x[:, 0:1, :], axis=1) # (batch_size, 1, output) , using for regression task (pool outputs by taking representation of first token)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits

    def compute_masked_loss(self, y, y_hat, mask):
        """ Computes loss between true y and prediction over masked locations"""
        # Select masked locations
        y_masked = tf.boolean_mask(y, mask)
        y_hat_masked = tf.boolean_mask(y_hat, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_masked, y_hat_masked, from_logits=True)
        return tf.reduce_mean(loss)

    def compute_masked_accuracy(self, y, y_hat, mask):
        m = tf.keras.metrics.CategoricalAccuracy()
        # Compute accuracy over masked tokens
        y_onehot = tf.one_hot(y, depth=self.vocab_size)
        y_masked = tf.boolean_mask(y_onehot, mask)
        y_hat_masked = tf.boolean_mask(y_hat, mask)

        # Compute accuracy
        m.update_state(y_masked, y_hat_masked)
        accuracy = m.result()
        return accuracy

    def compute_loss(self, y, y_hat):
        """ Computes loss, as mean absolute error, between true y and prediction.

        Args:
            y: true labels
            y_hat: output prediction

        Returns:
            absolute error
        """
        y_hat = y_hat # batch, last, 1
        y_hat = tf.cast(y_hat, tf.float64)
        error = tf.abs(y - y_hat)
        return tf.reduce_mean(error)

    def compute_rmse(self, y, y_hat, axis=None):
        """Computes the root mean squared error between true y and prediction.

        Args:
            y: true labels
            y_hat: output prediction

        Returns:
            mse
        """
        y_hat = y_hat
        error = (y - y_hat) ** 2
        return tf.sqrt(tf.reduce_mean(error, axis=axis))


### LSTM PREDICTOR ###
class RNNPredictor(tf.keras.Model):
    """ Predictor class to predict the specificity of peptide substrates."""

    def __init__(self, vocab_size, embedding_dim, rnn_units,
                 p, regu, max_len, output_len, mask_zero=False, training=True):
        super(RNNPredictor, self).__init__()
        self.training = training

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=max_len, mask_zero=mask_zero) #batch_input_shape=[batch_size, max_len])

        # LSTM
        # BLOCK 1: LSTM, DROP
        self.lstm = tf.keras.layers.LSTM(
            rnn_units, recurrent_regularizer=L2(regu), recurrent_dropout=p, return_sequences=True, stateful=False)
        self.lstm = tf.keras.layers.Bidirectional(self.lstm)
        self.drop = tf.keras.layers.Dropout(p)

        # BLOCK 2: LSTM, DROP
        self.lstm2 = tf.keras.layers.LSTM(
          rnn_units, recurrent_regularizer=L2(regu), recurrent_dropout=p, return_sequences=False, stateful=False)
        self.lstm2 = tf.keras.layers.Bidirectional(self.lstm2)
        self.drop2 = tf.keras.layers.Dropout(p)

        self.dense = tf.keras.layers.Dense(2 * rnn_units, activation='relu')
        self.dense_out = tf.keras.layers.Dense(output_len)

    def call(self, x):
        """ Forward pass through the model.

        Args:
            x: input to the model
        """
        x = self.embedding(x)
        x = self.lstm(x, training=self.training)
        if self.training:
            x = self.drop(x)
        x = self.lstm2(x, training=self.training)
        self.last_layer_embeddings = x
        if self.training:
            x = self.drop2(x)
        x = self.dense(x)
        x = self.dense_out(x)
        return x

    def compute_loss(self, y, y_hat):
        """ Computes loss, as mean absolute error, between true y and prediction.

        Args:
            y: true labels
            y_hat: output prediction

        Returns:
            absolute error
        """
        y_hat = y_hat #[:, -1, :] # batch, last, 1
        y_hat = tf.cast(y_hat, tf.float64)
        error = tf.abs(y - y_hat)
        return tf.reduce_mean(error)

    def compute_rmse(self, y, y_hat, axis=None):
        """Computes the root mean squared error between true y and prediction.

        Args:
            y: true labels
            y_hat: output prediction

        Returns:
            mse
        """
        y_hat = y_hat  # [:, -1, :]
        error = (y - y_hat) ** 2
        return tf.sqrt(tf.reduce_mean(error, axis=axis))


### LSTM GENERATOR MODELS ###
class RNNGenerator(tf.keras.Model):
    """ Generator class to learn how to predict new peptide substrates."""

    def __init__(self, batch_size, vocab_size, embedding_dim, rnn_units, p, regu, seq_len, training=True, num_layers=2):
        super(RNNGenerator, self).__init__()
        self.training = training
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # Embed Inputs
        self.embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim, batch_input_shape=[batch_size, seq_len])
        # NN
        self.lstm = [tf.keras.layers.LSTM(rnn_units, recurrent_regularizer=L2(regu), recurrent_dropout=p, return_sequences=True, stateful=True) for i in range(num_layers)]
        self.lstm = [tf.keras.layers.Bidirectional(self.lstm[i]) for i in range(num_layers)]
        self.drop = [tf.keras.layers.Dropout(p) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(2 * rnn_units, activation='relu')
        self.dense_out = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        """ Forward pass through the model.

        Args:
            x: input to the model
        """
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32)) # set scale of layers
        for i in range(self.num_layers):
            x = self.lstm[i](x, training=self.training)
            if self.training:
                x = self.drop[i](x)
        self.last_layer_embeddings = x
        x = self.dense(x)
        x = self.dense_out(x)
        return x

    def compute_masked_loss(self, y, y_hat, mask):
        """ Computes loss between true y and prediction over masked locations"""
        # Select masked locations
        y_masked = tf.boolean_mask(y, mask)
        y_hat_masked = tf.boolean_mask(y_hat, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_masked, y_hat_masked, from_logits=True)
        return tf.reduce_mean(loss)

    def compute_masked_accuracy(self, y, y_hat, mask):
        # Sample from predicted distribution to get characters
        m = tf.keras.metrics.CategoricalAccuracy()

        # Compute accuracy over masked tokens
        y_onehot = tf.one_hot(y, depth=self.vocab_size)
        y_masked = tf.boolean_mask(y_onehot, mask)
        y_hat_masked = tf.boolean_mask(y_hat, mask)
        # Compute accuracy
        m.update_state(y_masked, y_hat_masked)
        accuracy = m.result()
        return accuracy


class AutoregressiveRNN(tf.keras.Model):
    """ Generator class to generate new peptide substrates."""

    def __init__(self, batch_size, vocab_size, embedding_dim, rnn_units, p, regu, seq_len, training=True, mask_zero=False, num_layers=4):
        super(AutoregressiveRNN, self).__init__()
        self.training = training
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # Embed Inputs
        self.embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                                                   batch_input_shape=[batch_size, seq_len], mask_zero=mask_zero)
        # NN
        self.lstm = [tf.keras.layers.LSTM(rnn_units, recurrent_regularizer=L2(regu), recurrent_dropout=p,
                                         return_sequences=True, stateful=True) for i in range(num_layers)]
        self.drop = [tf.keras.layers.Dropout(p) for i in range(num_layers)]
        # Output layer: Dense (fully-connected) layer that transforms the LSTM output to vocab size
        self.dense = tf.keras.layers.Dense(2 * rnn_units, activation='relu')
        self.dense_out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        """ Forward pass through the model
        """
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))  # set scale of layers
        for i in range(self.num_layers):
            x = self.lstm[i](x, training=self.training)
            if self.training:
                x = self.drop[i](x)
        self.last_layer_embeddings = x
        x = self.dense(x)
        x = self.dense_out(x)
        return x

    def compute_loss(self, y, y_hat):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, y, y_hat):
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y, y_hat)
        return m.result()

def load_generator_model(model_type, training_scheme='unconditional', parent_dir=''):
    # Load relevant checkpoints, and model parameters
    batch_size = 1                                                                                                                                  

    if model_type == 'transformer':
        vocab_size = 22
        num_layers = 2
        num_heads = 6
        dropout = 0
        d_model = 64
        if training_scheme == 'unconditional':
            checkpoint_path = os.path.join(parent_dir+"weights/AUTOREG_transformer/unconditional/", "model.h5")
        elif training_scheme == 'conditional':
            num_layers=3
            checkpoint_path = os.path.join(parent_dir+"weights/AUTOREG_transformer/conditional/", "model.h5")
        elif training_scheme == 'both':
            num_layers=3
            checkpoint_path = os.path.join(parent_dir+"weights/AUTOREG_transformer/both/", "model.h5")
        elif training_scheme == 'rounded':
            num_layers=3
            d_model = 64
            checkpoint_path = os.path.join("weights/AUTOREG_transformer/both_rounded/", "model.h5") # rounded to tenth
        if training_scheme == 'unconditional':
            model = TransformerDecoder(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dff=d_model,  # dense params
                vocab_size=vocab_size,
                dropout_rate=dropout)
        else:
            model = cleavenet.models.ConditionalTransformerDecoder(
                                num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                dff=d_model, # dense params
                                vocab_size=vocab_size,
                                dropout_rate=dropout)

    elif model_type == 'lstm':
        regu = 0.01
        d_model = 64
        dropout = 0.2
        d_embed = 64
        seq_len = 11
        num_layers = 4
        vocab_size=22
        model = AutoregressiveRNN(batch_size, vocab_size, d_embed, d_model, dropout,
                                                   regu, seq_len, training=False, mask_zero=False, num_layers=num_layers)
        checkpoint_path = os.path.join("weights/AUTOREG_lstm/20231016-133250_GEN/", "model.h5")

    if training_scheme == 'unconditional':
        model.summary()
        model.load_weights(checkpoint_path)
    return model, checkpoint_path

def load_predictor_model(model_type, checkpoint_path, batch_size, mask_zero=False):
    d_model = 32
    len_mmps = 18

    if model_type == 'lstm':
        embedding_dim = 22
        dropout = 0.25
        regu = 0.01
        max_len = 10
        vocab_size = 21  # did not train with CLS token
        model = cleavenet.models.RNNPredictor(vocab_size, embedding_dim, d_model,
                                              dropout, regu, max_len, len_mmps, mask_zero=mask_zero)

    elif model_type == 'transformer':
        d_model = 32
        num_layers = 2
        num_heads = 6
        dropout = 0
        vocab_size = 22
        embedding_dim = 32
        model = cleavenet.models.TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim, #d_model,
            num_heads=num_heads,
            dff=d_model,  # dense params
            vocab_size=vocab_size,
            dropout_rate=dropout,
            output_dim=len_mmps,
            pool_outputs=True,
            mask_zero=mask_zero)

    fake_batch = np.array([[21, 20, 14, 8, 9, 13, 10 , 9 ,16, 17 , 3]])
    model(fake_batch, training=False) # build model in TF
    model.summary()
    model.load_weights(checkpoint_path)  # load weights
    return model


def inference(model, dataloader, causal=False, seq_len=10, penalty=1, verbose=False, conditioning_tag=None, temperature=1):
    if causal: # autoregressive inference
        if conditioning_tag is None:
            start_seq = np.array([dataloader.char2idx[dataloader.START]], dtype=np.int32)  # start from START token
            generated_seq = tf.expand_dims(start_seq, 0)
        else: 
            tag = np.array(conditioning_tag) # start from Z-scores
            generated_seq = (np.array([[]]), tag) # seq, tags
        reach_stop = False
        for i in range(seq_len):
            if reach_stop == False:
                prediction = model(generated_seq, training=False)
                predicted_id = tf.random.categorical(prediction[:, -1, :]/temperature, num_samples=1, dtype=tf.int32)
                if penalty > 1:
                    if conditioning_tag is None:
                        if predicted_id == generated_seq[0][i]:
                            penalized_prediction = prediction.numpy()
                            penalized_prediction[:, -1, int(predicted_id)] = prediction[0, -1, int(predicted_id)]/penalty
                            predicted_id = tf.random.categorical(penalized_prediction[:, -1, :], num_samples=1, dtype=tf.int32)
                    else:
                        if generated_seq[0].shape[1] > 0:
                            if predicted_id == generated_seq[0][0][i-1]:
                                penalized_prediction = prediction.numpy()
                                penalized_prediction[:, -1, int(predicted_id)] = prediction[0, -1, int(predicted_id)]/penalty
                                predicted_id = tf.random.categorical(penalized_prediction[:, -1, :], num_samples=1, dtype=tf.int32)
                    
                if conditioning_tag is None:
                    generated_seq = tf.concat([generated_seq, predicted_id], 1)
                else:
                    generated_seq = (np.expand_dims(np.append(generated_seq[0], predicted_id), 0), tag)   
                
                if verbose:
                    if conditioning_tag is None:    
                        print("step", i, "seq", cleavenet.data.untokenize_sequences(generated_seq, dataloader))
                    else:
                        print("step", i, "seq", cleavenet.data.untokenize_sequences(generated_seq[0].astype(int), dataloader))
                
                if predicted_id == dataloader.char2idx[dataloader.STOP]:
                    reach_stop = True
            else:
                break
        if conditioning_tag is None:
            generated_seq = generated_seq[0]
        else:
            generated_seq = generated_seq[0][0][:-1].astype(int)
    else: # MLM inference
        generated_seq = np.zeros((seq_len), dtype=int) + dataloader.char2idx[dataloader.MASK] # start from all mask token
        loc = np.arange(seq_len)
        np.random.shuffle(loc)  # shuffle order
        for i in loc:
            predictions = model(tf.expand_dims(generated_seq, 0), training=False)
            predicted_id = tf.random.categorical(predictions[:, i, :], num_samples=1)
            generated_seq[i] = predicted_id
            print("step",i, "out seq", cleavenet.data.untokenize_sequences(generated_seq, dataloader))

    return generated_seq


def prediction(data_path_kukreja, gen_data, generated_dir, true_zscores=None, true_mmps=None, checkpoint_dir='save/', predictor_model_type='transformer', number_top_candidates=50):
    if not os.path.exists(generated_dir):
        os.mkdir(generated_dir)
    if predictor_model_type == 'transformer':
        ensembles = ['transformer_0/',
                 'transformer_1/',
                 'transformer_2/',
                 'transformer_3/',
                 'transformer_4/'
                 ]
    elif predictor_model_type == 'lstm':
        print("USING LSTM PREDICTOR FOR EVALUATION")
        ensembles = [
            'lstm_0/',
            'lstm_1/',
            'lstm_2/',
            'lstm_3/',
            'lstm_4/'
        ]
    
    kukreja = cleavenet.data.DataLoader(data_path_kukreja, seed=0, task='regression', model=predictor_model_type, test_split=0.2,
                                        dataset='kukreja')
    
    max_seq_len = max([len(s) for s in gen_data])
    gen_data = [cleavenet.data.pad(seq, max_seq_len=max_seq_len, pad_token='-') for seq in gen_data]
    x_all = cleavenet.data.tokenize_sequences(gen_data, kukreja)
    if predictor_model_type == 'transformer':
        cls_idx = kukreja.char2idx[kukreja.CLS]
        x_all = np.stack([np.append(np.array(cls_idx), s) for s in x_all])
    num_samples = len(gen_data)
    batch_size = num_samples
    print(num_samples, gen_data[0])
    predictions = []
    for e_num, ensemble in enumerate(ensembles):
        print("Running", e_num, ensemble)
        print("EVALUATING SEQUENCES FROM", generated_dir)
        checkpoint_path = os.path.join(checkpoint_dir, ensemble, "model.h5")
        # Build and load predictor model
        model = cleavenet.models.load_predictor_model(model_type=predictor_model_type, checkpoint_path=checkpoint_path,
                                                      batch_size=batch_size, mask_zero=True)
        # Predict z-scores
        y_hat = model(x_all, training=False)  # forward pass
        if true_zscores is not None:
            subset_preds = []
            for i, m in enumerate(mmps):
                if m in true_mmps:
                    j = true_mmps.index(m)
                    subset_preds.append(y_hat[:, i])
            subset_preds = np.stack(subset_preds, axis=1)
            mae = tf.reduce_mean(tf.abs(true_zscores - subset_preds), axis=0)
            plotter.plot_mae(mae, true_mmps, generated_dir + str(e_num) + '_')
            rmse = model.compute_rmse(true_zscores, subset_preds, axis=0)
            plotter.plot_rmse(rmse, true_mmps, generated_dir + str(e_num) + '_')
            plotter.plot_parity(true_zscores, subset_preds, true_mmps, generated_dir)
        predictions.append(y_hat)
        if e_num == (len(ensembles)-1): # save embeddings from last ensemble model for plotting later
            embeddings = model.last_layer_embeddings
            np.save(os.path.join(generated_dir, 'embeddings.npy'), np.array(embeddings))

    predictions = np.stack(np.array(predictions))
    print(predictions.shape)
    means, std = analysis.confidence_score(predictions, mmps)

    if true_zscores is not None:
        print("plot true vs mean predicted")
        pred = []
        for i, m in enumerate(mmps):
            if m in true_mmps:
                print(m)
                j = true_mmps.index(m)
                pred.append(y_hat[:, i])
                scores = analysis.save_to_dataframe(x_all, true_zscores[:, j], i, means, std, z_cutoff=0,
                                                    write_top_scores=True,
                                                    dataloader=kukreja, mmp=m, save_path=generated_dir)
                plotter.true_pred_ranked_scatter_z(scores, generated_dir, m)
                plotter.confidence_ranked_scatter_z(scores, generated_dir, m)
                plotter.confusion(scores['Cleaved true'], scores['Cleaved pred'], generated_dir, m)
                plotter.plot_auc(true_zscores[:,j], y_hat[:,i], i, mmps, generated_dir)
        pred = np.stack(pred, axis=1)

    
    print("Calculated confidence")
    analysis.eval_all_mmp(means, x_all, kukreja, generated_dir, z_score_cutoff=0)
    
    for i,m in enumerate(mmps):
        if i == 0:
            if os.path.exists(os.path.join(generated_dir, 'weighted_all_scores.csv')):
                os.remove(os.path.join(generated_dir, 'weighted_all_scores.csv'))
            if os.path.exists(os.path.join(generated_dir, 'all_uncertainty.csv')):   
                os.remove(os.path.join(generated_dir, 'all_uncertainty.csv'))
            if os.path.exists(os.path.join(generated_dir, 'all_scores.csv')):   
                os.remove(os.path.join(generated_dir, 'all_scores.csv'))
        print("Iterating over", m)
        scores = analysis.save_to_dataframe(x_all, None, i, means, std, z_cutoff=0, write_top_scores=True,
                                            find_matches=True, dataloader=kukreja, mmp=m, save_path=generated_dir,
                                            top=number_top_candidates)
        plotter.confidence_ranked_scatter_z(scores, generated_dir, m)
        confidence_threshold = plotter.confidence_histogram(scores, generated_dir, m)
        plotter.confidence_ranked_scatter_z(scores, generated_dir, m, threshold=confidence_threshold)

    return means, std


def predict_scores_simple(substrates, checkpoint_dir='../weights/', save_dir='outputs/', model_architecture='transformer'):
    data_dir = cleavenet.utils.get_data_dir()
    data_path = os.path.join(data_dir, "kukreja.csv")
    pred_zscores, std_zscores = cleavenet.models.prediction(data_path,
                                                            substrates,
                                                            save_dir,
                                                            predictor_model_type=model_architecture,
                                                            checkpoint_dir=checkpoint_dir)
    return pred_zscores, std_zscores

def simple_inference(num_seqs, repeat_penalty, temperature):
    data_dir = cleavenet.utils.get_data_dir()
    data_path = os.path.join(data_dir, "kukreja.csv")
    kukreja = cleavenet.data.DataLoader(data_path, seed=0, task='generator', model='autoreg', test_split=0.2,
                                                dataset='kukreja')
    # From dataloader get necessary variables
    start_id = kukreja.char2idx[kukreja.START]
    # Load model
    model, checkpoint_path = cleavenet.models.load_generator_model(model_type='transformer',
                                                                   training_scheme='rounded')
    # Fake run to load data and build model
    conditioning_tag_fake = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    generated_seq = cleavenet.models.inference(model, kukreja, causal=True, seq_len=11,
                                               penalty=1, # no penalty
                                               verbose=False,
                                               conditioning_tag=conditioning_tag_fake,
                                               temperature=1 # no temp
                                               )
    conditioning_tag = [[start_id]] # unconditional generation
    untokenized_seqs = []
    for i in range(len(conditioning_tag)):
        for j in range(num_seqs):
            model.built=True
            model.load_weights(checkpoint_path)  # Load model weights
            # Generate using loaded weights
            generated_seq = cleavenet.models.inference(model, kukreja, causal=True, seq_len=11,
                                                       penalty=repeat_penalty,
                                                       verbose=False,
                                                       conditioning_tag=[conditioning_tag[i]], temperature=temperature)
            #tokenized_seqs.append(generated_seq)
            untokenized_seqs.append(''.join(kukreja.idx2char[generated_seq]))
    return untokenized_seqs