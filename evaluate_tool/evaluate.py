#!/usr/bin/env python
#coding:utf8

import copy
from string_tool import *
import sys
import json

main_path = '/'.join(sys.argv[0].split('/')[:-1])
if main_path:
    main_path += '/'

qid_answer_expand = {}

def format_string(string):
    string = strQ2B(string)
    string = string.lower()
    string = drop_punctuation(string)
    string = filter_blank(string)
    string = string.strip()
    return string

def load_qid_answer_expand(file_path):
    with open(main_path+file_path, "r") as file:
        for line in file:
            if len(line.strip().split("\t")) != 3:
                print line.strip()
            qid, answer, answer_expand = line.strip().split("\t")
            answer_expand = set(answer_expand.split("|"))
            tmp_answer_expand = copy.copy(answer_expand)
            for element in tmp_answer_expand:
                answer_expand.add(format_string(element))
            qid_answer_expand[qid] = (answer, answer_expand)

def is_exact_match_answer(qid, competitor_answer):
    if qid not in qid_answer_expand:
        raise ValueError("Invalid qid:%s" % qid)
    competitor_answer = competitor_answer.strip()
    if competitor_answer == "":
        return "0"
    format_competitor_answer = format_string(competitor_answer)
    answer, answer_expand = qid_answer_expand[qid]
    if format_competitor_answer in answer_expand:
        return "1"
    tmp_set1 = set([format_string(element) for element in drop_punctuation(competitor_answer).lower().split(" ")])
    tmp_set2 = set([format_string(element) for element in drop_punctuation(answer).lower().split(" ")])
    if tmp_set1 == tmp_set2:
        return "1"
    return "0"

def cacu_character_level_f(qid, competitor_answer):
    if qid not in qid_answer_expand:
        raise ValueError("Invalid qid:%s" % qid)
    competitor_answer = competitor_answer.strip()
    if competitor_answer == "":
        return (0.0, 0.0, 0.0, None)
    format_competitor_answer = format_string(competitor_answer)
    format_competitor_answer_tokens = split_string(format_competitor_answer)
    answer, answer_expand = qid_answer_expand[qid]
    max_f = 0.0
    max_f_precision = 0.0
    max_f_recall = 0.0
    max_f_answer = None
    for tmp_answer in answer_expand:
        tmp_answer_tokens = split_string(format_string(tmp_answer))
        tmp_answer_tokens_copy = copy.copy(tmp_answer_tokens)
        right_count = 0
        for format_competitor_answer_token in format_competitor_answer_tokens:
            if format_competitor_answer_token in tmp_answer_tokens_copy:
                right_count += 1
                tmp_answer_tokens_copy.remove(format_competitor_answer_token)
        if right_count == 0:
            continue
        precision = 1.0 * right_count / len(format_competitor_answer_tokens)
        recall = 1.0 * right_count / len(tmp_answer_tokens)
        f = 2 * precision * recall / (precision + recall)
        if f > max_f:
            max_f = f
            max_f_precision = precision
            max_f_recall = recall
            max_f_answer = tmp_answer
    return (max_f, max_f_precision, max_f_recall, max_f_answer)

def evaluate(input_file, output_file): 
    from contextlib import nested
    import sys
    load_qid_answer_expand("qid_answer_expand")
    total = 0
    right = 0
    sum_f = 0.0
    with nested(open(input_file, "r"), open(output_file, "w")) as (infile, outfile):
        for line in infile:
            total += 1
            items = line.replace("\n", "").split("\t")
            if len(items) != 2:
                raise ValueError("Invalid line: '%s', which should have 2 fields. The 2 fields are query_id and competitor_answer" % line.strip())
            qid, competitor_answer = items
            right_flag = is_exact_match_answer(qid, competitor_answer)
            if right_flag == "1":
                right += 1
            max_f, max_f_precision, max_f_recall, max_f_answer = cacu_character_level_f(qid, competitor_answer)
            sum_f += max_f
            outfile.write("%s\t%s\t%s\t%f\t%f\t%f\t%s\n" % (qid, competitor_answer, right_flag, max_f, max_f_precision, max_f_recall, max_f_answer))
    #print "query-level precision=%d/%d=%f" % (right, total, 1.0*right/total)
    #print "character-level average f value=%f/%f=%f" % (sum_f, total, sum_f/total)
    return 1.0*right/total,sum_f/total,(1.0*right/total+sum_f/total)/2.

if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1], sys.argv[2])))
