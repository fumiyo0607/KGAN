import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt

model_file = 'model_last-fm_epoch=300'
meta_file_name = 'weights-299.meta'

def _generate_transE_score( h, t, r):
    
    embeddings = tf.concat([user_embed, entity_embed], axis=0)
    embeddings = tf.expand_dims(embeddings, 1)

    h_e = tf.nn.embedding_lookup(embeddings, h)
    t_e = tf.nn.embedding_lookup(embeddings, t)
        
    # relation embeddings: batch_size * kge_dim
    r_e = tf.nn.embedding_lookup(relation_embed, r)

    '''
    
    # relation transform weights: batch_size * kge_dim * emb_dim
    trans_M = tf.nn.embedding_lookup(trans_W, r)

    # batch_size * 1 * kge_dim -> batch_size * kge_dim
    h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
    t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])
    
    '''

    # l2-normalize
    h_e = tf.math.l2_normalize(h_e, axis=1)
    r_e = tf.math.l2_normalize(r_e, axis=1)
    t_e = tf.math.l2_normalize(t_e, axis=1)

    kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)

    return kg_score


if __name__ == '__main__':
    
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./{}/{}'.format(model_file, meta_file_name))

    # モデルの復元
    saver.restore(sess,tf.train.latest_checkpoint('./' + model_file + '/'))
    # グラフを復元
    graph = tf.get_default_graph()

    user_embed = graph.get_tensor_by_name("user_embed:0")
    entity_embed = graph.get_tensor_by_name("entity_embed:0")
    relation_embed = graph.get_tensor_by_name("relation_embed:0")


    saved_path = '../Analysis/each_user_result/'
    file_name = 'last-fm_each_user_ret_epoch=0.pickle'

    with open (saved_path + file_name, 'rb') as f:
        result = pickle.load(f)

    print(result[0])



    user_rec_items = defaultdict()

    for user, ret in result.items():
        items = dict(ret['top_N_items'])
        items = items.keys()
    
        user_rec_items[user]  = list(items)

    kge_dim = 64

    kg_score = _generate_transE_score(1,2,3)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        sess.run(kg_score)
        kg_score = kg_score.eval()

        print(kg_score)