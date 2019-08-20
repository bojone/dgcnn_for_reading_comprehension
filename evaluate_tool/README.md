## 搜狗提供的评测脚本（只可用于SogouQA部分）

脚本使用说明（evaluate.py）

* evaluate.py接收两个命令行参数：（1） 第一个参数是输入文件（要求UTF8编码），格式为query_id\tcompetitor_answer；（2）第二个参数是输出文件（自己命名）。输出文件有7列，分别是query_id, competitor_answer, 是否正确，按字匹配的最大F1值，最大F1值对应precision，最大F1值对应recall，最大F1值时的答案。最后输出评测集上的平均正确率、F1值。
* evaluate.py运行中会加载qid_answer_expand文件，其中包含标准答案和其他正确答案。第一列是问题的query_id，第二列是我们发布的json数据集中的标准答案，第三列则是标准答案和出现在passage中的其他正确答案。
* 脚本读取选手给出的答案，并做大小写转换，全半角转换，去标点，去空格的归一化操作。
* 如果3中的归一化答案在qid_answer_expand的第三列中，则正确。F1根据字匹配计算，每个query取由标准答案和其他正确答案计算出的最大F1。
