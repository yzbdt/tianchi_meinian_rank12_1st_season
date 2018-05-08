import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import datetime
import os
os.chdir(os.path.dirname(os.getcwd()))

params = {
    'learning_rate': 0.025,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',  # 使用均方误差
    'num_leaves': 60,  # 最大叶子数for base learner
    'feature_fraction': 0.6,  # 选择部分的特征
    'min_data': 100,  # 一个叶子上的最少样本数
    'min_hessian': 1,  # 一个叶子上的最小 hessian 和，子叶权值需大于的最小和
    'verbose': 1,
    'lambda_l1': 0.3,  # L1正则项系数
    'device': 'cpu',
    'num_threads': 8, #最好设置为真实核心数
}
def pick_feature_map(feature):
    filename = './data/category_feature_map.npy'
    f_map = np.load(filename)
    f_map = (set(feature) & set(f_map)) - set(['vid'])
    return list(f_map)

def truncate(df):
    for i in df.columns[1:]:
        temp_df = df[i][df[i].notnull()].copy()
        llt = np.percentile(temp_df.values, 0.05)
        ult = np.percentile(temp_df.values, 99.95)
        temp_df[temp_df > ult] = ult
        temp_df[temp_df <= llt] = llt
        df[i] = temp_df
    return df

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred)
    return ('mse',score,False)

def offline_eval(label, output, flag): #要根据数据处理过程自己修改的
    return 0
    label = np.log(label+1)
    output = np.log(output+1)

    return mean_squared_error(label, output)

def offline_eval10(label, output, flag): #要根据数据处理过程自己修改的
    return 0
    label = np.log10(label+1)
    output = np.log10(output+1)

    return mean_squared_error(label, output)


def train(strs, count, train_dit, dit_count):
    print(count)
    params['learning_rate'] = train_dit['lr'][count]
    path = './data/'
    filename = 'feature_nums_df.csv'
    train_df = pd.read_csv(path + filename, encoding='gbk')
    filename = 'feature_nums2_df.csv'
    train_df2 = pd.read_csv(path + filename, encoding='gbk')
    train_df = train_df.merge(train_df2, how='inner', on='vid')

    filename = 'throw_nan_label_' + strs + '.csv'
    with open(path + filename, encoding='gbk') as f:
        label_df = pd.read_csv(f)

    train_df = truncate(train_df)

    filename = 'feature_short_str2.csv'
    category_df = pd.read_csv(path+filename, encoding='gbk')
    train_df = train_df.merge(category_df, how='inner', on='vid')

    filename = 'feature_d2v.csv'
    category_df2 = pd.read_csv(path+filename, encoding='gbk')
    train_df = train_df.merge(category_df2, how='inner', on='vid')

    filename = './data/meinian_round1_test_b_20180505.csv'
    test_df = pd.read_csv(filename, encoding='gbk')
    s = pd.DataFrame(test_df['vid'].copy())
    test_df = s.merge(train_df, how='inner', on='vid')
    submission = np.zeros(test_df.shape[0])
    feature = train_df.columns[1:]
    if not train_dit['is_start']:
        filename = 'feature_importance.csv'
        feature = pd.read_csv(path + filename, encoding='gbk').T
        feature = feature[0]
        thrh = np.percentile(feature.values, train_dit['feature_decay'][dit_count])
        feature = feature[feature>thrh].index
    train_df = train_df.merge(label_df, how='inner', on='vid')
    categorical_feature = pick_feature_map(feature)

    kfo2 = KFold(5, shuffle=True)
    t0 = time.time()
    loss = 0
    times = 5
    offline_loss10 = 0
    offline_loss = 0
    for j in range(times):
        train_index, valid_index = next(kfo2.split(train_df))
        print('训练{}ing...第{}次'.format(strs, j+1))
        train_data2 = train_df.iloc[train_index]
        vali_data = train_df.iloc[valid_index]
        train_data_set = lgbm.Dataset(train_data2[feature], train_data2[strs])
        vali_data_set = lgbm.Dataset(vali_data[feature], vali_data[strs])
        model = lgbm.train(params, train_data_set, num_boost_round=3000, valid_sets=vali_data_set, verbose_eval=100,
                               feval=None, early_stopping_rounds=100,  categorical_feature=categorical_feature)
        print('CV训练已经用时{}秒'.format(time.time() - t0))
        train_pred = model.predict(vali_data[feature])
        loss = loss + mean_squared_error(vali_data[strs], train_pred)
        offline_loss += offline_eval(vali_data[strs], train_pred, count)
        offline_loss10 += offline_eval10(vali_data[strs], train_pred, count)
        submission += model.predict(test_df[feature])
        feat_imp = pd.Series(model.feature_importance(), index=feature).sort_values(ascending=False)
    return feat_imp, loss/times, offline_loss/times, offline_loss10/times, submission/times

def train_head():
    target_lst = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    for p in range(1):
        total_loss = 0
        filename = './data/meinian_round1_test_b_20180505.csv'
        last_df = pd.read_csv(filename, encoding='gbk')
        count = 0
        for i in target_lst:
            train_dit = {
                'feature_decay': [5, 35, 5, 5, 5, 5, 5],
                'last_lost': [1],
                'is_start': True,
                'lr': [0.018, 0.018, 0.03, 0.04, 0.04]

            }

            path = './data/'
            filename = './data/meinian_round1_test_b_20180505.csv'
            test_df = pd.read_csv(filename, encoding='gbk')
            for i_b in range(8):
                i_c = i_b - 1
                if i_c >= 0:
                    train_dit['is_start'] = False
                f_imp, loss_single, loss_single_offline, loss_single_offline10, submission = train(i, count, train_dit,
                                                                                                   i_c)
                loss = loss_single
                test_df[i] = np.exp(submission) - 1
                feature_imp = pd.DataFrame(f_imp).T
                if i_b == 0:
                    all_fea = test_df[i]
                    all_fea_loss = loss_single
                if (loss < train_dit['last_lost'][i_b]) or i_b <= 1:
                    feature_imp.to_csv(path + 'feature_importance.csv', index=False)
                    train_dit['last_lost'].append(loss)
                    last_df[i] = test_df[i]
                else:
                    if train_dit['last_lost'][-1] <= all_fea_loss:
                        print('参数{}的预测值的范围为，'.format(i), test_df[i].describe())
                        print('5折CV损失-mse', train_dit['last_lost'])
                        total_loss += train_dit['last_lost'][i_b]
                        print(i, '进行到', i_b - 1, ' 丢弃的模型的损失为 ', loss)
                    else:
                        print('还是全特征最好')
                        print('总体mse变化', train_dit['last_lost'])
                        print('5折CV损失-mse', all_fea_loss)
                        last_df[i] = all_fea
                        total_loss += all_fea_loss
                    break
                if (i_b >= 7):
                    print('参数{}的预测值的范围为，'.format(i), test_df[i].describe())
                    print('5折CV损失-mse', train_dit['last_lost'])
                    total_loss += train_dit['last_lost'][i_b+1]
                    print(i, '进行到', i_b - 1)
                    break
            count += 1
        print('线下损失', total_loss / 5)
        last_df.to_csv('./submit/submit_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                       index=False, header=False)