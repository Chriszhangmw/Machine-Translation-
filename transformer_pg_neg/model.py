
import logging
import tensorflow as tf
from tqdm import tqdm

from data_load import Vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, noam_scheme
from utils import convert_idx_to_token_tensor, split_input

logging.basicConfig(level=logging.INFO)
import numpy as np
class Transformer:
    def __init__(self, hp):
        self.hp = hp
        self.train_mode = hp['mode']
        vocab = Vocab(hp['vocab_path'],hp['vocab_size'])
        self.token2idx = vocab.word2id
        self.idx2token = vocab.id2word
        embedding = np.loadtxt(hp['sif_embedding_path'], dtype=np.float32)
        self.embeddings = tf.Variable(embedding,
                                       dtype=tf.float32,
                                       trainable=False)
    def encode(self, xs,training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.x, sents1 = xs
            # self.f, _ = fre

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, self.x) # (N, T1, d_model)
            #embedding_sif
            enc = tf.svd(enc)[1]
            # print('&' * 20)
            print(enc.get_shape())
            #
            s, u, v = tf.linalg.svd(enc)
            print(s.get_shape())
            print(u.get_shape())
            print(v.get_shape())
            print('&' * 20)


            enc *= self.hp['enc_units']**0.5 # scale

            enc += positional_encoding(enc, self.hp['max_enc_len'])
            enc = tf.layers.dropout(enc, 0.1, training=training)
            ## Blocks
            for i in range(self.hp['num_blocks']):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc, _ = multihead_attention(queries=enc,
                                                  keys=enc,
                                                  values=enc,
                                                  num_heads=self.hp['num_heads'],
                                                  dropout_rate=0.1,
                                                  training=training,
                                                  causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp['attn_units'], self.hp['enc_units']])
        self.enc_output = enc
        return self.enc_output, sents1

    def decode(self, xs,fre,ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        self.memory = memory
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            self.decoder_inputs, y, sents2 = ys
            x, _, = xs
            frequency,_= fre
            # print('x is ::::::::::::::',x)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)  # (N, T2, d_model)
            # print('&' * 20)
            print(dec.get_shape())
            #
            s, u, v  = tf.linalg.svd(dec)
            print(s.get_shape())
            print(u.get_shape())
            print(v.get_shape())
            dec = tf.matrix_transpose(v)
            print(v.get_shape())
            # print('&'*20)
            # print(dec.get_shape())

            dec *= self.hp['enc_units'] ** 0.5  # scale

            dec += positional_encoding(dec, self.hp['max_dec_len'])

            before_dec = dec

            dec = tf.layers.dropout(dec, 0.1, training=training)

            attn_dists = []
            # Blocks
            for i in range(self.hp['num_blocks']):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec, _ = multihead_attention(queries=dec,
                                                 keys=dec,
                                                 values=dec,
                                                 num_heads=self.hp['num_heads'],
                                                 dropout_rate=0.1,
                                                 training=training,
                                                 causality=True,
                                                 scope="self_attention")
                    # Vanilla attention
                    dec, attn_dist = multihead_attention(queries=dec,
                                                          keys=self.memory,
                                                          values=self.memory,
                                                          num_heads=self.hp['num_heads'],
                                                          dropout_rate=0.1,
                                                          training=training,
                                                          causality=False,
                                                          scope="vanilla_attention")
                    attn_dists.append(attn_dist)
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp['attn_units'], self.hp['enc_units']])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)

        with tf.variable_scope("gen", reuse=tf.AUTO_REUSE):
            gens = tf.layers.dense(tf.concat([before_dec, dec, attn_dists[-1]], axis=-1), units=1, activation=tf.sigmoid,
                                   trainable=training, use_bias=False)
        logits = tf.nn.softmax(logits) # shape is (?, 50, 50000)
        # print('&'*30)
        # print('attn_dists shape is ',len(attn_dists))
        # print('attn_dists[-1] is ',attn_dists[-1])
        # print('&' * 30)
        # print('logits shape is :',logits.get_shape())
        # print('&' * 30)

        # final distribution
        # print('x shape is :',x.get_shape())
        # print('gens shape is :',gens.get_shape())
        self.logits = self._calc_final_dist(x, gens, logits, attn_dists[-1],frequency)
        # x shape is x shape is : (?, 400)
        # gens shape is : (?, 50, 1)
        #logits shape is (?, 50, 50000)
        #attn_dists[-1] shape is (?, 50, 400)


        return self.logits, y, sents2

    def _calc_final_dist(self, x, gens, vocab_dists, attn_dists,frequency):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          x: encoder input which contain oov number
          gens: the generation, choose vocab from article or vocab
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                       The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution', reuse=tf.AUTO_REUSE):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = gens * vocab_dists
            attn_dists = (1-gens) * attn_dists

            batch_size = tf.shape(attn_dists)[0]

            # print('$'*30)
            # print('frequency shape is :',frequency.get_shape())
            # print('$' * 30)

            uniform_batch = []
            # for _ in batch_size:
            #     x =
            #
            #     negative_weights = np.zeros((400, 400))




            dec_t = tf.shape(attn_dists)[1]
            # dec_t shape is : ()
            # print('dec_t shape is :',dec_t.get_shape())
            attn_len = tf.shape(attn_dists)[2]

            dec = tf.range(0, limit=dec_t) # [dec]
            dec = tf.expand_dims(dec, axis=-1) # [dec, 1]
            dec = tf.tile(dec, [1, attn_len]) # [dec, atten_len]
            dec = tf.expand_dims(dec, axis=0) # [1, dec, atten_len]
            dec = tf.tile(dec, [batch_size, 1, 1]) # [batch_size, dec, atten_len]

            x = tf.expand_dims(x, axis=1) # [batch_size, 1, atten_len]    考虑在这里加负采样？？？？
            #  Tensor("split:0", shape=(?, 400), dtype=int32)  这里的x也就是ID， 所以考虑在这里加  没问题吧
            # shape is (?, 1, 400)
            # print('&' * 30)
            # print('x shape is :', x.get_shape())
            # print('&' * 30)
            x = tf.tile(x, [1, dec_t, 1]) # [batch_size, dec, atten_len]
            x = tf.stack([dec, x], axis=3)
            # x shape is (?, ?, 400, 2)
            # print('&' * 30)
            # print('x shape is :', x.get_shape())
            # print('&' * 30)


            # for i in range(400):
            #     negative_weights[i][i] = uniform[i]
            # negative_weights = tf.convert_to_tensor(negative_weights)
            #
            # print('@'*40)
            # print('attn_dists 乘以之前的维度：', attn_dists.get_shape())
            if self.train_mode == 'train':

                negative_weights = tf.matrix_diag(frequency)
                attn_dists = tf.expand_dims(attn_dists, axis=-1)
                attn_dists = tf.reshape(attn_dists,[-1,50,1,400])
                # print('negative_weights shape is ',negative_weights.get_shape())
                negative_weights = tf.reshape(negative_weights,[-1,1,400,400])
                attn_dists = tf.multiply(attn_dists, negative_weights)
                attn_dists = tf.matrix_diag_part(attn_dists)
            print('attn_dists 乘以之后的维度：',attn_dists.get_shape())
            # print('@' * 40).linalg.diag



            attn_dists_projected = tf.map_fn(fn=lambda y: tf.scatter_nd(y[0], y[1], [dec_t, self.hp['vocab_size']]),
                                             elems=(x, attn_dists), dtype=tf.float32)
            #  attn_dists is Tensor("final_distribution/mul_1:0", shape=(?, 50, 400), dtype=float32, device=/device:GPU:0)
            # 就应该是在这里加，50-400的矩阵，每行代表解码的一个字，与400个字也就是400列的关系，用这个关系再乘以本身各个列的权重，作为最后
            #该解码字与输入各个字的关系
            # print('attn_dists is :',attn_dists)
            # attn_dists_projected shape is : (?, 50, 50000)
            # print('&' * 30)
            # print('attn_dists_projected shape is :', attn_dists_projected.get_shape())
            # print('&' * 30)

            final_dists = attn_dists_projected + vocab_dists # final_dists shape is : (?, 50, 50000)
            # print('final_dists shape is :',final_dists.get_shape())

        return final_dists

    def _calc_loss(self, targets, final_dists):
        """
        calculate loss
        :param targets: reference
        :param final_dists:  transformer decoder output add by pointer generator
        :return: loss
        """
        with tf.name_scope('loss'):
            dec = tf.shape(targets)[1]
            batch_nums = tf.shape(targets)[0]
            dec = tf.range(0, limit=dec)
            dec = tf.expand_dims(dec, axis=0)
            dec = tf.tile(dec, [batch_nums, 1])
            indices = tf.stack([dec, targets], axis=2) # [batch_size, dec, 2]

            loss = tf.map_fn(fn=lambda x: tf.gather_nd(x[1], x[0]), elems=(indices, final_dists), dtype=tf.float32)
            loss = tf.log(0.9) - tf.log(loss)

            nonpadding = tf.to_float(tf.not_equal(targets, self.token2idx["PAD"]))  # 0: <pad>
            loss = tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

            return loss

    def train(self, xs,fre, ys):
        """
        train model
        :param xs: dataset xs
        :param ys: dataset ys
        :return: loss
                 train op
                 global step
                 tensorflow summary
        """
        tower_grads = []
        global_step = tf.train.get_or_create_global_step()
        global_step_ = global_step * 1
        lr = noam_scheme(self.hp['attn_units'], global_step_, self.hp['warmup_steps'])
        optimizer = tf.train.AdamOptimizer(lr)
        losses = []
        xs,fre, ys = split_input(xs,fre, ys, 1)
        # with tf.variable_scope(tf.get_variable_scope()):
        for no in range(1):
            with tf.device("/gpu:0"):

                memory, sents1 = self.encode(xs[no])
                logits, y, sents2 = self.decode(xs[no],fre[no],ys[no], memory)
                # tf.get_variable_scope().reuse_variables()

                loss = self._calc_loss(y, logits)
                losses.append(loss)
                grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)
                grads = self.average_gradients(tower_grads)
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
                loss = sum(losses) / len(losses)
                tf.summary.scalar('lr', lr)
                tf.summary.scalar("train_loss", loss)
                summaries = tf.summary.merge_all()

        return loss, train_op, global_step_, summaries,lr

    def average_gradients(self, tower_grads):
        """
        average gradients of all gpu gradients
        :param tower_grads: list, each element is a gradient of gpu
        :return: be averaged gradient
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
