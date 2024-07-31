import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from utils import read_json_jsonl_file
import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np

# we modified this script from https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py

# SPLIT_TOKEN="\t----- SplitToken -----\t"

IMPUTE_EMPTY="PLACE_HOLDER"

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):

    try:
        res = func_timeout(meta_time_out, execute_sql,
                                args=(predicted_sql, ground_truth, db_place))
        info=""
    except KeyboardInterrupt:
        sys.exit(0)

    except FunctionTimedOut:
        info = [(f'timeout',)]
        res = 0


    except Exception as e:
        info = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0

    result = {'sql_idx': idx, 'res': res, "info":info}
    return result


def package_sqls(sql_path:str, 
                db_root_path: str, # 
                mode='pred', 
                skip_1st_row_for_gpt=True,
                sql_key='SQL'):
    clean_sqls = []
    db_path_list = []
    # GPT prediction file
    if mode == 'pred':
        predict_sql_list: List[Dict[str, str]] = read_json_jsonl_file(sql_path)
        for idx, record in enumerate(predict_sql_list):

            # first record is some meta data in the setting of this project.
            if skip_1st_row_for_gpt and idx == 0:
                continue
            db_id = record['db_id']
            pred_sql = record[sql_key]
            if pred_sql == "":
                pred_sql = IMPUTE_EMPTY          
            clean_sqls.append(pred_sql)
            db_path_list.append(os.path.join(db_root_path, db_id, db_id+".sqlite"))
    elif mode == 'gold':
        gold_sql_list: List[Dict[str, str]] = read_json_jsonl_file(sql_path)
        for idx, record in enumerate(gold_sql_list):
            db_id = record['db_id']
            pred_sql = record[sql_key]            
            clean_sqls.append(pred_sql)
            db_path_list.append(os.path.join(db_root_path, db_id, db_id+".sqlite"))
    else:
        raise KeyError(f"Unknown mode: {mode}, the mode should be set as 'pred' or 'gold'")
    return clean_sqls, db_path_list



def run_sqls_parallel(sqls, db_places, num_cpus=10, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in tqdm(enumerate(sqls)):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
        # result = execute_model(predicted_sql, ground_truth, db_places[i], i, meta_time_out)
        # print(result)
        # exec_result.append(result)

    pool.close()
    pool.join()

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_difficult(exec_results:List[Dict[str, str]], diff_json_path=None):

    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    if diff_json_path is not None:
        contents = load_json(diff_json_path)
    else:
        # set default level to simple if diff_json_path is None
        contents = [ {"difficulty":"simple"} for _ in range(len(exec_results))
                    ]

        
    simple_results, moderate_results, challenging_results = [], [], []

    for i,content in enumerate(contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    if len(simple_results)>0:
        simple_acc = sum([res['res'] for res in simple_results])/len(simple_results)
    else:
        simple_acc = np.nan
    
    if len(moderate_results)>0:
        moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results)
    else:
        moderate_acc = np.nan
    if len(challenging_results)>0:
        challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results)
    else:
        challenging_acc=np.nan

    all_acc = sum(results)/num_queries
    executable_rate = sum( [1 if x['info']=='' else 0 for x in exec_results] ) / num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100,  count_lists, executable_rate*100



def print_data(score_lists,count_lists, ex_rate):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

    print('======================================    Executable  =====================================')
    print(np.round(ex_rate,3))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument("--pred_key", type=str, required=True, default='predict_query')
    args_parser.add_argument("--gold_key", type=str, required=True, default="SQL")

    args_parser.add_argument('--num_cpus', type=int, default=20) # processes is the number of worker processes to use. If processes is None then the number returned by os.cpu_count() is used.
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--diff_json_path',type=str, default=None)

    args = args_parser.parse_args()
    exec_result = []


    pred_queries, db_paths = package_sqls(sql_path=args.predicted_sql_path, 
                                            db_root_path=args.db_root_path, 
                                            mode="pred",
                                            skip_1st_row_for_gpt=True,
                                            sql_key=args.pred_key#"predict_query"
                                            )
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(sql_path=args.ground_truth_path, 
                                            db_root_path=args.db_root_path,
                                            mode="gold", # gt
                                            skip_1st_row_for_gpt=False,
                                            sql_key=args.gold_key
                                            )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    print('start calculate metrics')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists, ex_rate = \
        compute_acc_by_difficult(exec_result,args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists,count_lists, ex_rate)
    print('===========================================================================================')
    print("Finished evaluation")
    