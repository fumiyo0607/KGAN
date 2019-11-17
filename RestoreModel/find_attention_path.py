import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.sparse as sp
from tqdm import tqdm


def get_entity_to_entity_attention_score(e_h, e_t):

    e_h_idx = e_h + len(users)
    e_t_idx = e_t + len(users)
    att = attention_score[e_h_idx, e_t_idx]

    return att

def get_user_to_item_attention_score(user_id, item_id):

    e_h_idx = user_id
    e_t_idx = item_id + len(users)
    att = attention_score[e_h_idx, e_t_idx]

    return att

rec_item_path = defaultdict(dict)

for user in tqdm(users):

    rec_item_list = user_rec_items[user]
    interacted_items = list(dict.fromkeys(all_data[str(user)]))

    for rec_item_id in rec_item_list:

        rec_item_entity_relation_df = kg_att_df[kg_att_df['e_h'] == rec_item_id]

        entity_list = list(rec_item_entity_relation_df['e_t'])

        for entity_id in entity_list:

            target_rec_item_entity_relation_df = rec_item_entity_relation_df[rec_item_entity_relation_df['e_t']==entity_id]
    
            relation_list    = list(target_rec_item_entity_relation_df['r'])

            for entity_to_rec_item_r in relation_list:

                # attention score : entity to recommended item
                entity_to_rec_item_att = get_entity_to_entity_attention_score(entity_id, rec_item_id)

                item_entity_relation_df = kg_att_df[kg_att_df['e_t'] == entity_id]
                head_items = list(item_entity_relation_df['e_h'])

                interacted_item_entity_relation_df = item_entity_relation_df.query('e_h in {}'.format(interacted_items))

                if len(interacted_item_entity_relation_df) > 0:

                    e_h_list      = list(interacted_item_entity_relation_df['e_h'])
                    e_t_list      = list(interacted_item_entity_relation_df['e_t'])
                    relation_list = list(interacted_item_entity_relation_df['r'])

                    for e_h, e_t, r in zip(e_h_list, e_t_list, relation_list):
                        
                        item_id          = e_h  # interacted item
                        item_to_entity_r = r    # relation between interacted item and entity

                        # consider item to entity to rec_item path
                        if item_to_entity_r == entity_to_rec_item_r:
                            
                            # attention score : interacted item to entity
                            item_to_entity_att = get_entity_to_entity_attention_score(item_id, entity_id)
                    
                            # attention score : user to interacted item
                            user_to_item_att = get_user_to_item_attention_score(user, item_id)

                            total_att_score = user_to_item_att + item_to_entity_att + entity_to_rec_item_att

                            if !( rec_item_id in rec_item_path[user]) or ( total_att_score > max_att) :

                                max_att = total_att_score 

                                rec_item_path[user][rec_item_id] =  {

                                                        'total_att_score'    : total_att_score,
                                                        'relation'           : item_to_entity_r,
                                                        'item_id'            : item_id, 
                                                        'entity_id'          : entity_id, 
                                                        'rec_item_id'        : rec_item_id,
                                                        'user_to_item'       : user_to_item_att,
                                                        'item_to_entity'     : item_to_entity_att,
                                                        'entity_to_rec_item' : entity_to_rec_item_att
                                                    }

                                path_counter += 1
                            





        






