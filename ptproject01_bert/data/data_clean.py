import numpy as np
import pandas as pd
import re
REMOVE_WORDS = ['|', '[', ']', '语音', '图片']

def parse_data(train_path, test_path):
    # 读取csv
    train_df = pd.read_csv(train_path, encoding='utf-8')
    # 去除report为空的
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    # 剩余字段是输入，包含Brand,Model,Question,Dialogue，如果有空，填充即可
    train_df.fillna('', inplace=True)
    # 实际的输入X仅仅选择两个字段，将其拼接起来
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = []
    if 'Report' in train_df.columns:
        train_y = train_df.Report
        assert len(train_x) == len(train_y)

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y



def save_data(data_1, data_2, data_path_1):
    train = zip(data_1,data_2)
    with open(data_path_1,'w',encoding='utf-8') as f1:
        count_1 = 0
        for line_x,line_y  in train:
            if isinstance(line_x,str):
                line_x = line_x.replace('[图片]',"")
                line_x = line_x.replace('[语音]', "")
                line_x = line_x.replace('|', "")
                line_x = line_x.replace('\t', "")
                line_x = line_x.replace(' ', "")
                # line_x = ''.join([s for s in line_x if s not in REMOVE_WORDS])
                f1.write(line_x + 'fengefu' + line_y)
                f1.write('\n')
                count_1 += 1
        print('train length is ',count_1)

    # with open(data_path_2,'w',encoding='utf-8') as f2:
    #     count_2 = 0
    #     for line in data_2:
    #         if isinstance(line,str):
    #             seg_list =  segment(line.strip(),cut_type='word')
    #             #考虑去除special symbol
    #             seg_list = [word for word in seg_list if word not in stopwords]
    #             if len(seg_list) > 0:
    #                 seg_line = ' '.join(seg_list)
    #                 f2.write('%s' % seg_line)
    #                 f2.write('\n')
    #             else:
    #                 f2.write('随时 联系' + '\n')
    #             count_2 += 1
    #
    #     print('train y length is ',count_2)
    #
    # with open(data_path_3, 'w', encoding='utf-8') as f3:
    #     count_3 = 0
    #     for line in data_3:
    #         if isinstance(line, str):
    #             seg_list = segment(line.strip(), cut_type='word')
    #             # 考虑去除special symbol
    #             seg_list = [word for word in seg_list if word not in stopwords]
    #             if len(seg_list) > 0:
    #                 seg_line = ' '.join(seg_list)
    #                 f3.write('%s' % seg_line)
    #                 f3.write('\n')
    #                 count_3 += 1
    #     print('test x length is ', count_3)




if __name__ == '__main__':
    train_x,train_y,test_x,_ = parse_data('./AutoMaster_TrainSet.csv','./AutoMaster_TestSet.csv')

    save_data(train_x,train_y,'./data_train/train.csv')
