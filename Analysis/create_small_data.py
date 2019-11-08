import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np

'''
*************************
Config
*************************
'''
data_set = 'last-fm'
user_num = 100

kg_df = pd.read_csv('./data/{}/kg_final.txt'.format(data_set), sep=' ', header=None, names=['e_h', 'r', 'e_t'])

'''
*************************
entity data の読み込み
*************************
'''

org_id_list = []
remap_id_list = []

f = open('./data/{}/entity_list.txt'.format(data_set))
line = f.readline()

while line:
    data = line.strip()
    data_list = data.split(' ')

    org_id = data_list[0]
    remap_id = data_list[1]

    org_id_list.append(org_id)
    remap_id_list.append(remap_id)

    line = f.readline()

f.close()

org_id_list.pop(0)
remap_id_list.pop(0)

e_df = pd.DataFrame(
    data={
        'org_id'   : org_id_list,
        'remap_id' : remap_id_list
    },
    columns=['org_id', 'remap_id']
)



'''
************************************************
*************************
kg_df の entity を freebase_id を使って書き換え 
************************************************
'''

e_map = defaultdict(str)

org_ids = list(e_df['org_id'])
remap_ids = list(e_df['remap_id'])

for org_id, remap_id in zip(org_ids, remap_ids):
    e_map[remap_id] = org_id

def convert_remap_id_to_org_id(remap_id):
    remap_id = str(remap_id)
    return e_map[remap_id]

# kg_df using free_base id
kg_df['e_h'] = kg_df['e_h'].map(convert_remap_id_to_org_id)
kg_df['e_t'] = kg_df['e_t'].map(convert_remap_id_to_org_id)


'''
*************************
item を freebase_id を使って書き変えるための準備
*************************
'''

item_df = pd.read_csv('./data/{}/item_list.txt'.format(data_set), sep=' ')

i_map = defaultdict(str)

freebase_ids = item_df['freebase_id']
remap_ids = item_df['remap_id']

for freebase_id, remap_id in zip(freebase_ids, remap_ids):
    remap_id = str(remap_id)
    i_map[remap_id] = freebase_id

def convert_item_id_to_freebase_id(item_id):
    return i_map[item_id]


'''
*************************
train.txt のデータを 指定サイズ (user_num) で分割し， item を freebase_id を使って書き換え 
*************************
'''

train_data = defaultdict(list)
f = open('./data/{}/train.txt'.format(data_set))
line = f.readline()

user_count = 0

while line :

    data = line.strip()
    data_list = data.split()

    user = data_list[0]
    items = data_list[1:]
    train_data[user] = items
    
    line = f.readline()
    
f.close()

all_train_user_list = []
all_train_item_list = []

for u, items in train_data.items():
    for i in items:
        all_train_user_list.append(u)
        all_train_item_list.append(i)

short_train_user_list = []
short_train_item_list = []

short_user_data_count = 0

for u, items in train_data.items():
    if user_count < user_num:
        for i in items:
            short_train_user_list.append(u)
            short_train_item_list.append(i)
            short_user_data_count += 1
        user_count += 1
    else:
        break

delete_train_user_list = all_train_user_list[short_user_data_count:]
delete_train_item_list = all_train_item_list[short_user_data_count:]

# all data
all_train_df = pd.DataFrame(
    data = {
        'user_id' : all_train_user_list,
        'item_id' : all_train_item_list
    },
    columns=['user_id', 'item_id']
)

# target user num data
short_user_train_df = pd.DataFrame(
    data = {
        'user_id' : short_train_user_list,
        'item_id' : short_train_item_list
    },
    columns=['user_id', 'item_id']
)

# delete user data
delete_user_train_df = pd.DataFrame(
    data = {
        'user_id' : delete_train_user_list,
        'item_id' : delete_train_item_list
    },
    columns=['user_id', 'item_id']
)

# item_id を free_base_id で書き換える
all_train_df['item_id']         = all_train_df['item_id'].map(convert_item_id_to_freebase_id)
short_user_train_df['item_id']  = short_user_train_df['item_id'].map(convert_item_id_to_freebase_id)
delete_user_train_df['item_id'] = delete_user_train_df['item_id'].map(convert_item_id_to_freebase_id)


'''
*************************
test.txt のデータを 指定サイズ (user_num) で分割し， item を freebase_id を使って書き換え 
*************************
'''

test_data = defaultdict(list)
f = open('./data/{}/test.txt'.format(data_set))
line = f.readline()

user_count = 0

while line :

    data = line.strip()
    data_list = data.split()

    user = data_list[0]
    items = data_list[1:]
    train_data[user] = items
    
    line = f.readline()
    
f.close()

all_test_user_list = []
all_test_item_list = []

for u, items in train_data.items():
    for i in items:
        all_test_user_list.append(u)
        all_test_item_list.append(i)

short_test_user_list = []
short_test_item_list = []

short_user_data_count = 0

for u, items in train_data.items():
    if user_count < user_num:
        for i in items:
            short_test_user_list.append(u)
            short_test_item_list.append(i)
            short_user_data_count += 1
        user_count += 1
    else:
        break

delete_test_user_list = all_test_user_list[short_user_data_count:]
delete_test_item_list = all_test_item_list[short_user_data_count:]

# all data
all_test_df = pd.DataFrame(
    data = {
        'user_id' : all_test_user_list,
        'item_id' : all_test_item_list
    },
    columns=['user_id', 'item_id']
)

# target user num data
short_user_test_df = pd.DataFrame(
    data = {
        'user_id' : short_train_user_list,
        'item_id' : short_train_item_list
    },
    columns=['user_id', 'item_id']
)

# delete user data
delete_user_test_df = pd.DataFrame(
    data = {
        'user_id' : delete_test_user_list,
        'item_id' : delete_test_item_list
    },
    columns=['user_id', 'item_id']
)

# item_id を free_base_id で書き換える
all_test_df['item_id']         = all_test_df['item_id'].map(convert_item_id_to_freebase_id)
short_user_test_df['item_id']  = short_user_test_df['item_id'].map(convert_item_id_to_freebase_id)
delete_user_test_df['item_id'] = delete_user_test_df['item_id'].map(convert_item_id_to_freebase_id)


'''
*************************
kg_df から delete したデータに含まれる (h, r, t) のtriple を 削除 
*************************
'''
short_user_df  = pd.concat( [short_user_train_df, short_user_test_df] )
delete_user_df = pd.concat( [delete_user_test_df, delete_user_train_df] )

short_user_item_list  = list(short_user_df['item_id'].unique())
delete_user_item_list = list(delete_user_df['item_id'].unique())

print('kg_df から削除するアイテムのリストを作成')

delete_item_list = delete_user_item_list

for item in tqdm(short_user_item_list):
    if item in delete_item_list:
        delete_item_list.remove(item)

print('kg_dfからdelete_item_listを含むtripleを削除')

_e_h_list = kg_df['e_h']
_r_list   = kg_df['r']
_e_t_list = kg_df['e_t'] 

delete_index = []
index_counter = 0

# TODO: ここがめっちゃ時間かかる 
for e_h, r, e_t in zip(tqdm(_e_h_list), _r_list, _e_t_list):
    if (e_h in delete_item_list) or (e_t in delete_item_list):
        delete_index.append(index_counter)
    index_counter += 1

# numpy へ変換
_e_h_np = np.array(_e_h_list)
_e_t_np = np.array(_e_t_list)
_r_np   = np.array(_r_list)

_e_h_np = np.delete(_e_h_np, delete_index)
_e_t_np = np.delete(_e_t_np, delete_index)
_r_np   = np.delete(_r_np, delete_index)

e_h_list = _e_h_np.tolist()
r_list   = _r_np.tolist()
e_t_list = _e_t_np.tolist()


short_user_kg_df = pd.DataFrame(
    data = {
        'e_h' : e_h_list,
        'r'   : r_list,
        'e_t' : e_t_list
    },
    columns = ['e_h', 'r', 'e_t']
)

print('reshaping knowledge graph done .. !!')

'''
************************
short_user のデータの entity list を作成 
************************
'''

# include both items and side info
all_entity_list = list(pd.concat([short_user_kg_df['e_h'], short_user_kg_df['e_h']]))
all_entity_list = set(all_entity_list)

# only side info not including items 
print('creating entity list which is not includeing items ')
entity_list = [i for i in tqdm(all_entity_list) if i not in short_user_item_list]

short_user_all_entity_list = short_user_item_list + entity_list

short_user_entity_map = defaultdict(int)
entity_counter = 0

for e in short_user_all_entity_list:
    short_user_entity_map[e] = entity_counter
    entity_counter += 1

'''
************************
short_user のデータの free_base_id を振り直す
************************
'''

def convert_free_base_id_to_short_user_entity_id(free_base_id):
    return short_user_entity_map[free_base_id]

# train data
short_user_test_df['item_id'] = short_user_test_df['item_id'].map(convert_free_base_id_to_short_user_entity_id)
# test data
short_user_train_df['item_id'] = short_user_train_df['item_id'].map(convert_free_base_id_to_short_user_entity_id)
# knowlage graph data
short_user_kg_df['e_h'] = short_user_kg_df['e_h'].map(convert_free_base_id_to_short_user_entity_id)
short_user_kg_df['e_t'] = short_user_kg_df['e_t'].map(convert_free_base_id_to_short_user_entity_id)


'''
************************
short_user のデータを吐き出す
************************
'''
short_test_user_id_list = list(short_user_test_df['user_id'])
short_test_item_id_list = list(short_user_test_df['item_id'])

# test data
short_test_data = defaultdict(list)

for user, item in zip( short_test_user_id_list, short_train_item_list):
    short_test_data[user].append(item)

# train data
short_train_data = defaultdict(list)

for user, item in zip( short_train_user_list, short_train_item_list):
    short_train_data[user].append(item)

# knowledge graph
short_user_kg_e_h = list(short_user_kg_df['e_h'])
short_user_kg_r   = list(short_user_kg_df['r'])
short_user_kg_e_t = list(short_user_kg_df['e_t'])


# ファイルへ書き出し

file_path = './small_data/{}_usernum={}'.format(data_set, user_num)

if not os.path.exists(file_path): os.makedirs(file_path)

train_file      = file_path + '/train.txt'
test_file       = file_path + '/test.txt'
kg_final_file   = file_path + '/kg_final.txt'

with open(train_file, mode='a') as f:

    for user, items in short_train_data.items():
        line = []
        line.append(user)

        for i in items:
            line.append(i)
    
        f.write(' '.join(line))
        f.write('\n')


with open(test_file, mode='a') as f:

    for user, items in short_test_data.items():
        line = []
        line.append(user)

        for i in items:
            line.append(i)
    
        f.write(' '.join(line))
        f.write('\n')

with open(kg_final_file, mode='a') as f:

    for e_h, r, e_t in zip(short_user_kg_e_h, short_user_kg_r, short_user_kg_e_t):
        line = []

        line.append(str(e_h))
        line.append(str(r))
        line.append(str(e_t))
    
        f.write(' '.join(line))
        f.write('\n')

print('All Done ...!')