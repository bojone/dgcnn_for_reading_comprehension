#coding:utf8

import re
import sys

blank_regexp = re.compile(r'\s+')
punctuation = set()

main_path = '/'.join(sys.argv[0].split('/')[:-1])
if main_path:
    main_path += '/'

with open(main_path+"punctuation", "r") as file:
    for line in file:
        punctuation.add(line.strip().decode("utf8"))

def drop_punctuation(string, codec="utf8"):
    """删除所有标点符号"""
    ustring = string.decode(codec, "ignore");
    rstring = ""
    for uchar in ustring:
        if uchar not in punctuation:
            rstring += uchar
        else:
            rstring += " "
    return rstring.encode(codec, 'ignore')

def split_string(string, codec="utf8"):
    split_tokens = []
    ustring = string.decode(codec, "ignore");
    for uchar in ustring:
        split_tokens.append(uchar.encode(codec, "ignore"))
    return split_tokens

def strQ2B(string, codec="utf8"):
    """全角转半角"""
    ustring = string.decode(codec, "ignore")
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring.encode(codec, "ignore")

def strB2Q(string, codec="utf8"):
    """半角转全角"""
    ustring = string.decode(codec, "ignore")
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化                  
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring.encode(codec, "ignore")

def filter_blank(string):
    return blank_regexp.sub('', string)

def filter_extra_blank(string):
    return blank_regexp.sub(' ', string)
