import pandas as pd
import numpy as np
import os
import time
import re
import collections
import jieba
jieba.load_userdict('wordbook.txt')
import gensim
from gensim.models import Doc2Vec

def re_sub(pat, s):
    c = re.sub(pat, ' ', s).strip().split()
    if c:
        return c[0]
    else:
        return c


def nums_log(temp_df, strs):
    temp = np.log(temp_df[strs].astype('float')+1)
    temp_df[strs] = temp
    return temp_df


def split(ss):
    in_str = str(ss)
    if in_str == 'nan':
        return ss
    else:
        return ' '.join(jieba.cut(ss))


def single_feat2vec(df, feat, vector_size=5):
    text_pattern = './data/d2v.txt'
    wait_play_original = df[feat].copy()
    wait_play = wait_play_original.dropna()
    wait_play = wait_play.apply(split)
    wait_play.to_csv(text_pattern, encoding='utf-8', sep='\n', index=False)

    # doc2vec parameters
    window_size = 5
    min_count = 1
    sampling_threshold = 0
    negative_size = 0
    train_epoch = 10
    dm = 0  # 0 = dbow; 1 = dmpv
    worker_count = 16  # number of parallel processes

    docs = gensim.models.doc2vec.TaggedLineDocument(text_pattern)
    model_doc = Doc2Vec(docs, vector_size=vector_size, window=window_size,
                        min_count=min_count, sample=sampling_threshold,
                        workers=worker_count, hs=0, dm=dm,
                        negative=negative_size, dbow_words=1, dm_concat=1, epochs=train_epoch)

    def str2vec(ss):
        list_ss = list(model_doc.infer_vector(ss.split(sep=' ')))
        list_ss_str = [str(num) for num in list_ss]
        return '/'.join(list_ss_str)

    wait_play = wait_play.apply(str2vec)
    wait_play_original[~wait_play_original.isnull()] = wait_play

    out_ones = wait_play_original.str.split('/', expand=True)
    new_index = []
    for i in range(vector_size):
        new_index.append('{}_{}'.format(feat, i))
    out_ones.columns = new_index
    return out_ones


def creat_label():
    path = './data/'
    filename = 'meinian_round1_train_20180408.csv'
    llt = [30, 10, 0.01, 0.01, 0.01]
    ult = [270, 200, 30, 20, 20]
    df_train = pd.read_csv(path+filename, encoding='gbk')
    lst = list(df_train.columns)[1:]
    for i in range(5):
        header = ['vid']
        header.append(lst[i])
        temp_df = df_train[header].copy()
        l = temp_df.shape[0]
        strs = lst[i]
        data_pattern = path + 'throw_nan_label_' + strs + '.csv'
        if os.path.exists(data_pattern):
           continue
        for j in range(l):
            try:
                d = float(df_train[strs][j])
                if d>=ult[i] or d<=llt[i]:
                    temp_df = temp_df.drop(j)
            except:
                try:
                    c = str(df_train[strs][j]).strip('<>+-=轻度乳糜')
                    c = float(c)
                    temp_df[strs][j] = c
                    if c >= ult[i] or c <= llt[i]:
                        temp_df = temp_df.drop(j)
                except:
                    try:
                        temp_df = temp_df.drop(j)
                    except:
                        pass
        temp_df = nums_log(temp_df, strs)
        temp_df.to_csv('./data/throw_nan_label_' + strs + '.csv', index=False)


def read_raw_data():
    path = './data/'
    data_pattern = path + 'raw_feature.csv'
    if os.path.exists(data_pattern):
        return
    counter = 0
    temp_lst = [[], [], []]
    filename = 'meinian_round1_data_part1_20180408.txt'
    with open(path + filename, 'r', encoding='UTF-8-sig') as infile:
        columns = infile.readline().strip().split('$')
        while True:
            line = infile.readline().strip().split('$')
            if line == ['']:
                break
            for i in range(3):
                temp_lst[i].append(line[i])
            counter += 1

    filename = 'meinian_round1_data_part2_20180408.txt'
    temp_lst2 = [[], [], []]
    with open(path + filename, 'r', encoding='UTF-8-sig') as infile:
        columns2 = infile.readline().strip().split('$')
        while True:
            line = infile.readline().strip().split('$')
            if line == ['']:
                break
            for i in range(3):
                temp_lst2[i].append(line[i])
            counter += 1

    dit = {columns[i]: temp_lst[i] for i in range(3)}
    check_df1 = pd.DataFrame(dit)
    dit2 = {columns2[i]: temp_lst2[i] for i in range(3)}
    check_df2 = pd.DataFrame(dit2)
    set1 = set(check_df1['table_id'])
    set2 = set(check_df2['table_id'])
    same_table_id = set2 & set1
    print(same_table_id)
    del check_df1, check_df2, temp_lst, temp_lst2

    counter = 0
    temp_dit = {}
    filename = 'meinian_round1_data_part1_20180408.txt'
    with open(path + filename, 'r', encoding='UTF-8-sig') as infile:
        columns = infile.readline().strip().split('$')
        while True:
            line = infile.readline().strip().split('$')
            if line == ['']:
                break
            if line[0] in temp_dit:
                if line[1] in temp_dit[line[0]]:
                    try:
                        c = float(line[2])
                        temp_dit[line[0]][line[1]] = line[2]
                    except:
                        if line[2]:
                            temp_dit[line[0]][line[1]] = str(temp_dit[line[0]][line[1]]) + ',' + str(line[2])
                else:
                    temp_dit[line[0]][line[1]] = line[2]
            else:
                temp_dit[line[0]] = {line[1]: line[2]}
            counter += 1

    check_df = pd.DataFrame(temp_dit)
    check_df = check_df.T
    print(check_df.columns)

    filename = 'meinian_round1_data_part2_20180408.txt'
    with open(path + filename, 'r', encoding='UTF-8-sig') as infile:
        columns = infile.readline().strip().split('$')
        while True:
            line = infile.readline().strip().split('$')
            if line == ['']:
                break
            if line[1] in same_table_id:
                line[1] += 'A'
            if line[0] in temp_dit:
                if line[1] in temp_dit[line[0]]:
                    try:
                        c = float(line[2])
                        temp_dit[line[0]][line[1]] = line[2]
                    except:
                        if line[2]:
                            temp_dit[line[0]][line[1]] = str(temp_dit[line[0]][line[1]]) + ',' + str(line[2])
                else:
                    temp_dit[line[0]][line[1]] = line[2]
            else:
                temp_dit[line[0]] = {line[1]: line[2]}
            counter += 1

    check_df = pd.DataFrame(temp_dit)
    check_df = check_df.T
    print(check_df.columns)
    check_df2 = check_df.reset_index()
    lst = list(check_df2.columns)
    lst[0] = 'vid'
    check_df2.columns = lst
    print(check_df2.columns)
    check_df2.to_csv('./data/raw_feature.csv', encoding='gbk', index=False)


def simple_nums_feature():
    path = './data/'
    data_pattern = path + 'feature_nums_df.csv'
    if os.path.exists(data_pattern):
        return
    filename = 'raw_feature.csv'

    train_df = pd.read_csv(path+filename, encoding='gbk', low_memory=False)
    print('数据载入成功')

    feature_num = train_df.shape[1]
    print('特征数量：', feature_num)

    train_df.replace(['未查', '弃查'], np.nan, inplace=True)
    print('丢弃了弃查未查项')

    #这里丢弃了空缺值超过57109项的特征，共丢弃了1900+项
    missing_df = train_df.notnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'count']
    missing_df = missing_df.sort_values(by='count', ascending=True)
    missing_df = missing_df[missing_df['count']>200]
    missing_df = missing_df[missing_df['count'] < 57298]
    missing_df = ['vid'] + list(missing_df['column_name'])
    train_df = train_df[missing_df]
    feature_num = train_df.shape[1]
    print('特征数量：', feature_num)

    #数一数每一项的unique值有多少，丢弃只有一项的
    cat_dit = {}
    for i in train_df.columns[1:]:
        c = train_df[i]
        c = c[c.notnull()]
        c = c.unique().shape[0]
        cat_dit.update({i: c})
    cat_dit = pd.Series(cat_dit).sort_values(ascending=False)
    lst_str = ['vid']+list(cat_dit[cat_dit > 1].index)
    train_df = train_df[lst_str]
    print(len(lst_str))
    count = 0
    lst = ['vid']
    lst_str = ['vid']
    for i in train_df.columns[1:]:
        if count % 50 == 0:
            print(count, '/', feature_num)
        try:
            train_df[i] = train_df[i].astype('float')
            lst.append(i)
        except:
            lst_str.append(i)
        count += 1
    print(len(lst_str))
    print(len(lst))
    nums_df = train_df[lst]
    str_df = train_df[lst_str]

    str_df.to_csv(path + 'feature_str_df.csv', index=False)
    nums_df.to_csv(path + 'feature_nums_df.csv', index=False)


def more_nums_feature():
    path = './data/'
    data_pattern = path + 'feature_nums_waiteforprocess_df.csv'
    if os.path.exists(data_pattern):
        return

    filename = 'feature_str_df.csv'
    train_df = pd.read_csv(path + filename, encoding='gbk', low_memory=False)
    print('数据载入成功', '特征数量', train_df.shape[1])
    times = 0
    dit = {}
    for i in train_df.columns[1:]:
        times +=1
        if times % 50 == 0:
            print(times)
        count = 0
        notnull_num = train_df[i].notnull().sum()
        for j in range(train_df.shape[0]):
            try:
                c = float(train_df[i][j])
                if c is np.nan:
                    pass
                else:
                    count+=1
            except:
                pass
        dit.update({i: count/notnull_num})
    dit = pd.Series(dit).sort_values(ascending=False)
    print(dit.describe())
    str_df = train_df[['vid'] + list(dit[dit<=0.05].index)]
    num_df = train_df[['vid'] + list(dit[dit>=0.5].index)]
    temp = dit[dit<0.5]
    temp = temp[temp>0.05]
    other_df = train_df[['vid'] + list(temp.index)]

    str_df.to_csv('./data/feature_str2_df.csv', index=False)
    other_df.to_csv('./data/feature_unknown_df.csv', index=False)
    num_df.to_csv('./data/feature_nums_waiteforprocess_df.csv', index=False)


def more_nums_feature2():
    path = './data/'
    data_pattern = path + 'feature_nums2_df.csv'
    if os.path.exists(data_pattern):
        return

    filename = 'feature_nums_waiteforprocess_df.csv'

    df = pd.read_csv(path + filename, encoding='gbk', low_memory=False)
    print('数据载入成功', '特征数量', df.shape[1])
    pat = [re.compile(r'\d+月\d+日'), re.compile(r'[^\d.]'), re.compile(r'\.$')]

    cout = 0
    t0 = time.time()
    for i in df.columns[1:]:
        cout+=1
        lst = []
        if cout % 10 == 0:
            print('每10个用时{}秒'.format(time.time() - t0))
            print(cout, '/', df.shape[1])
            t0 = time.time()
        for j in range(df[i].shape[0]):
            c = str(df[i][j])
            if c == 'nan' or (not c) or (c is np.nan):
                continue
            for k in range(len(pat)):
                if c:
                    c = re_sub(pat[k], c)
                else:
                    break
            df[i][j] = c
            try:
                c = float(df[i][j])
            except:
                pass
    for i in df.columns[1:]:
        df[i] = pd.to_numeric(df[i], errors='coerce')

    df.to_csv('./data/feature_nums2_df.csv', index=False)


def wash_str():
    path = './data/'
    data_pattern = path + 'feature_str3_df.csv'
    if os.path.exists(data_pattern):
        return
    filename = 'feature_str2_df.csv'

    train_df = pd.read_csv(path + filename, encoding='gbk', low_memory=False)
    print('数据载入成功', '特征数量', train_df.shape[1])

    p = re.compile(r'[a-zA-Z0-9<>/,.，。()（）:/*% 、""“”‘’°？?=~;；：:]')
    word_lst = set(['未见异常', '未见明显异常', '正常', '未见异常未见异常', '正常正常', '正常正常正常', '未闻及异常',
                '正常正常正常正常正常', '未发现异常', '未见异常未见异常未见异常', '未见明显异常未见异常', '未发现明显异常',
                '未发现异常未发现异常', '正常正常正常正常', '无', '未见', '未见未见', '阴性', '阴性-', '阴性阴性', '-阴性',
                   '阴性阴性-', '阴性-阴性-', '抗体阴性', '-', '-~', '--'])
    count = 0
    t0 = time.time()
    for i in train_df.columns[1:]:
        count += 1
        if count % 10 == 0:
            print('每10个用时{}秒'.format(time.time() - t0))
            t0 = time.time()
            print(count, '/', train_df.shape[1])
        for j in range(train_df[i].shape[0]):
            if str(train_df[i][j]) == 'nan' or (not train_df[i][j]) or (train_df[i][j] is np.nan):
                continue
            train_df[i][j] = re.sub(p, '', train_df[i][j]).strip()
            if train_df[i][j] in word_lst:
                train_df[i][j] = '正常'

    train_df.to_csv('./data/feature_str3_df.csv', index=False)


def cut_str_feature():
    path = './data/'
    data_pattern = path + 'feature_long_str.csv'
    if os.path.exists(data_pattern):
        return
    filename = 'feature_str3_df.csv'

    train_df = pd.read_csv(path + filename, encoding='gbk', low_memory=False)
    nums, nums2 = train_df.shape
    count = 0
    max_length = 0
    lst =[]
    dit = collections.Counter()
    len_dit = {}
    for i in train_df.columns[1:]:
        max_i = 0
        if count % 10 ==0:
            print(count, '/', nums2)
        for j in range(train_df[i].shape[0]):
            c = list(jieba.cut(str(train_df[i][j]).encode('utf-8')))
            lst.append(c)
            max_i = max(len(c), max_i)
            len_dit.update({i:max_i})
            dit.update(c)

        count += 1

    df = pd.Series(dit).sort_values(ascending=False)
    len_df = pd.Series(len_dit).sort_values(ascending=False)
    print('cut over')
    print('max_length', max_length)
    print(len_df.describe())
    print(len_df.head(5))
    feature_lst = ['vid'] + list(len_df[len_df<6].index)
    short_str_df = train_df[feature_lst]
    values_dit = []
    unique_nums = []
    strange = []
    for i in short_str_df.columns[1:]:
        temp = short_str_df[i]
        u_nums = (temp.value_counts() > 1).sum()
        unique_nums.append(u_nums)
        if u_nums<=10:
            values_dit += list(temp.unique())
        else:
            strange.append(i)

    unique_nums = pd.Series(unique_nums)
    unique_nums.describe()

    short_str = list(set(short_str_df.columns[1:]) - set(['0213']))
    long_str = list(set(train_df.columns[1:]) - set(short_str))
    np.save(path + 'category_feature_map.npy', short_str)
    short_str = ['vid'] + short_str
    long_str = ['vid'] + long_str
    short_str_df = train_df[short_str]
    long_str_df = train_df[long_str]

    short_str_df.to_csv(path+'feature_short_str.csv', index=False)
    long_str_df.to_csv(path + 'feature_long_str.csv', index=False)


def cat_feature():
    path = './data/'
    data_pattern = path + 'feature_short_str2.csv'
    if os.path.exists(data_pattern):
        return

    filename = 'feature_short_str.csv'

    short_str_df = pd.read_csv(path+filename, encoding='gbk', low_memory=False)
    p = re.compile(r'[度]')
    temp = short_str_df['30007'].copy()
    for i in range(short_str_df['30007'].shape[0]):
        if i % 10000 == 0:
            print(i)
        if temp[i] == 'nan' or (not temp[i]) or temp[i]==np.nan:
            continue
        temp[i] = str(temp[i])
        temp[i] = re.sub(p, '', temp[i]).strip()
    short_str_df['30007'] = temp

    for i in short_str_df.columns[1:]:
        temp = short_str_df[i].copy()
        flag = temp.value_counts()>=5
        for j in range(short_str_df.shape[0]):
            if str(temp[j]) == 'nan' or not temp[j] or temp[j]==np.nan:
                continue
            if not flag[temp[j]]:
                temp[j] = np.nan
        short_str_df[i] = temp
        if len(temp.unique()) == 1:
            short_str_df.drop(i, axis=1, inplace=True)

    print('encoding')
    for i in short_str_df.columns[1:]:
        temp = short_str_df[i].copy()
        cate = list(temp.unique())
        value_dit = {cate[p]:p for p in range(len(cate))}
        for j in range(short_str_df.shape[0]):
            temp[j] = value_dit[temp[j]]

        short_str_df[i] = temp

    short_str_df.to_csv(path+'feature_short_str2.csv', index=False)


def long_str_feature():
    path = './data/'
    data_pattern = path + 'feature_d2v.csv'
    if os.path.exists(data_pattern):
        return

    filename = 'feature_long_str.csv'
    long_str_df = pd.read_csv(path + filename, encoding='gbk', low_memory=False)
    df = long_str_df['vid']
    long_str_df.drop('0102', axis=1, inplace=True)  # 整理的时候发现落下了0102，但加进来发现线下分数下降了，很奇怪，所以去掉试试
    count = 0
    for i in long_str_df.columns[1:]:
        if count % 10 == 0:
            print(count, '/', long_str_df.shape[1])
        temp_df = single_feat2vec(long_str_df, i)
        df = pd.concat([df, temp_df], axis=1)
        count += 1
    df.to_csv(path+'feature_d2v.csv', index=False)


