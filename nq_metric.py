'''
Implements scoring and metric for Google's Natural Questions (NQ) task
'''

import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def create_short_answer(answer_type, short_answer_score, short_answers, threshold=3.0):    
    if answer_type == 0:
        return ''
    elif answer_type == 1:
        return 'YES'
    elif answer_type == 2:
        return 'NO'
    elif short_answer_score < threshold:
        return ''
    else:
        answer = []
        for short_answer in short_answers:
            if short_answer["start_token"] != -1:
                answer.append(f"{short_answer['start_token']}:{short_answer['end_token']}")
        return " ".join(answer)

def create_long_answer(answer_type, long_answer_score, long_answer, threshold=3.0):
    if answer_type == 0:
        return ''
    elif long_answer_score < threshold:
        return ''
    elif long_answer['start_token'] == -1:
        return ''
    else:
        return f"{long_answer['start_token']}:{long_answer['end_token']}"


def get_test_df(nq_pred_dict, long_answer_threshold, short_answer_threshold):
    example_id_list = []
    answer_type_list = []
    long_answer_list = []
    short_answer_list = []
    long_answer_score_list = []
    short_answer_score_list = []
    for k, v in nq_pred_dict.items():
        example_id_list.append(int(k))
        
        answer_type = int(v['answer_type'])
        answer_type_list.append(answer_type)
        
        long_answer_score = v['long_answer_score']
        long_answer_score_list.append(long_answer_score)
        short_answer_score = v['short_answer_score']
        short_answer_score_list.append(short_answer_score)

        long_answer = create_long_answer(answer_type, long_answer_score, v['long_answer'], long_answer_threshold)
        long_answer_list.append(long_answer)
        
        short_answer = create_short_answer(answer_type, short_answer_score, v['short_answers'], short_answer_threshold)
        short_answer_list.append(short_answer)
             
    return pd.DataFrame({'example_id':example_id_list,
                         'answer_type':answer_type_list,
                         'long_answer':long_answer_list,
                         'short_answer':short_answer_list,
                         'long_answer_score':long_answer_score_list,
                         'short_answer_score':short_answer_score_list})

def get_labelled_df(jsonl_file):
    with open(jsonl_file, 'r') as f:
        example_id_list = []
        long_answer_list = []
        short_answers_list = []
        for line in f:
            json_dict = json.loads(line)
            example_id_list.append(int(json_dict['example_id']))
            
            a = json_dict['annotations'][0]
            long_answer = f"{a['long_answer']['start_token']}:{a['long_answer']['end_token']}" if a['long_answer']['start_token'] != -1 else ''
            long_answer_list.append(long_answer)
            
            short_answers = ''
            for sa in a['short_answers']:
                short_answers += f" {sa['start_token']}:{sa['end_token']}"
            short_answers_list.append(short_answers)
            
    return pd.DataFrame({'example_id':example_id_list, 'long_answer':long_answer_list, 'short_answers':short_answers_list})

def in_shorts(row):
    return row['short_answer'] in row['short_answers']

def get_f1(answer_df, label_df):
    short_label = (label_df['short_answers'] != '').astype(int)
    long_label = (label_df['long_answer'] != '').astype(int)

    long_predict = np.zeros(answer_df.shape[0])
    long_predict[(answer_df['long_answer'] == label_df['long_answer']) & (answer_df['long_answer'] != '')] = 1
    long_predict[(label_df['long_answer'] == '') & (answer_df['long_answer'] != '')] = 1  # false positive

    short_predict = np.zeros(answer_df.shape[0])
    short_predict[(label_df['short_answers'] == '') & (answer_df['short_answer'] != '')] = 1  # false positive
    a = pd.concat([answer_df[['short_answer']],label_df[['short_answers']]], axis = 1)
    a['short_answers'] = a['short_answers'].apply(lambda x: x.split())
    short_predict[a.apply(lambda x: in_shorts(x), axis = 1) & (a['short_answer'] != '')] = 1

    long_f1 = f1_score(long_label.values,long_predict)
    short_f1 = f1_score(short_label.values,short_predict)
    micro_f1 = f1_score(np.concatenate([long_label,short_label]),np.concatenate([long_predict,short_predict]))
    return micro_f1, long_f1, short_f1
