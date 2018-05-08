# Author: Li Shaofeng
import sys
sys.path.append('./code')
from preprocess import *
from train_model import *


if __name__=='__main__':
    print('数据预处理')
    # 标签数据
    print('标签数据')
    creat_label()

    # 读入原始数据生成raw_feature
    print('读入原始数据生成raw_feature')
    read_raw_data()

    # 获取纯数值特征
    print('获取纯数值特征')
    simple_nums_feature()

    # 清洗混入了少量字符的数值特征
    print('清洗混入了少量字符的数值特征')
    more_nums_feature()
    more_nums_feature2()

    # 清洗字符特征
    print('清洗字符特征')
    wash_str()

    # 切分为长字符和短字符特征
    print('切分为长字符和短字符特征')
    cut_str_feature()

    # 编码并生成catogory特征
    print('编码并生成catogory特征')
    cat_feature()

    # 生成长字符串特征
    print('生成长字符串特征')
    long_str_feature()

    print('特征处理完成，开始训练……')

    # 调用训练模块，训练模型并预测结果，由于采用随机抽取部分特征的模式，结果有随机性波动，
    # 单次训练耗时约2小时（cpu: Ryzen1700 OC 3.40Hz）
    train_head()

    print('训练完成')

