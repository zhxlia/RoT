import os
import re
import sys
import ast
import json
import random
import argparse

sys.path.append('.')

from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
from rouge_score import rouge_scorer
from typing import List, Dict, Any, Union, Tuple
random.seed(42)

PROMPT_ONE = '''Your task is to think step by step by traversing the given table to solve the question. 
Note that: 
1. You must traverse the whole table correctly and extract the relevant information.
2. Represent your answer with: \"Answer: <Your Answer>\".
Here is an example:

---

| Parish | Locality | Parish Priest | Founded | Closed |
|:---|:---|:---|:---|:---|
| St Mary | Bacup | Fr Frank Thorpe | 1852 | ---- |
| Our Immaculate Mother & St Anselm | Whitworth | Fr Frank Thorpe | 1860 | ---- |
| St Joseph | Stacksteads | ---- | 1947 | 2005 |
| St Joseph & St Peter | Newchurch-In-Rossendale | Fr Philip Boast | 1915 | ---- |
| The Immaculate Conception | Haslingden | Fr Canon John Mackie | 1854 | ---- |
| St Veronica (Chapel of Ease) | Helmshore | Served from The Immaculate Conception | 1959 | ---- |
| St James the Less | Rawtenstall | Fr David Lupton, Rural Dean | 1828 | ---- |

Question:
what's the number of parishes founded in the 1800s?

Solution:
To answer the question, I need to figure out how many parishes were founded in the 1800s based on the given table. Let me look at the table again. The table has several columns: Parish, Locality, Parish Priest, Founded, and Closed. My task is to count how many parishes were established in the 1800s.

First, I should understand what the "Founded" column represents. It likely indicates the year the parish was established. So, I need to look at each row and check the year under the "Founded" column. If the year is in the 1800s, I'll count that parish.

Let me go through each row one by one.

1. St Mary, Bacup: Founded in 1852. That's in the 1800s, so that's one.
2. Our Immaculate Mother & St Anselm, Whitworth: Founded in 1860. Also in the 1800s, so that's two.
3. St Joseph, Stacksteads: The "Founded" column is blank. Hmm, maybe it's not applicable or the data isn't provided. I'll skip this one for now.
4. St Joseph & St Peter, Newchurch-In-Rossendale: Founded in 1915. That's in the early 1900s, so not in the 1800s.
5. The Immaculate Conception, Haslingden: Founded in 1854. That's another one, so three.
6. St Veronica (Chapel of Ease), Helmshore: Founded in 1959. That's in the 1900s, so not relevant.
7. St James the Less, Rawtenstall: Founded in 1828. That's in the 1800s, so four.

So the answer is 4.

Answer: 4

---

Based on the above example, you need to traverse the below table to answer the question.

[Table]

Question:
[QUESTION]

Solution:
'''


PROMPT_ONE_HT = '''Your task is to think step by step by traversing the given table to solve the question. 
Note that: 
1. You must traverse the whole table correctly.
2. Represent your answer with: "Answer: <Your Answer>", without extra units (such as %) and explanations.
Here is an example:

---

## sex and marital status by fols of workers in the agricultural sector aged 15 years and over, three agricultural regions of new brunswick, 2011
| - | agricultural region 1 | agricultural region 1 | agricultural region 3 | agricultural region 3 | agricultural region 4 | agricultural region 4 |
|:---|:---|:---|:---|:---|:---|:---|
| | french-language workers | english-language workers | french-language workers | english-language workers | french-language workers | english-language workers |
| | percent | percent | percent | percent | percent | percent |
| sex |  |  |  |  |  | |
| female | 35.3 | 28.0 | 41.8 | 30.6 | 35.9 | 26.6 |
| male | 64.7 | 72.0 | 58.2 | 69.4 | 64.1 | 73.4 |
| marital status |  |  |  |  |  | |
| single | 18.7 | 24.8 | 26.9 | 26.1 | 25.6 | 32.8 |
| married | 51.3 | 53.9 | 47.8 | 56.7 | 54.4 | 57.8 |
| common-law | 17.1 | 10.9 | 22.4 | 6.4 | 11.8 | 7.8 |
| separated, divorced, or widowed | 12.8 | 10.6 | 0.0 | 10.8 | 7.7 | 0.0 |

Question:
which group of people has more workers in the agricultural secotr in 2011, regardless of the region or language? male or female?

Solution:
Okay, so I need to figure out whether there are more male or female workers in the agricultural sector in 2011 based on the given table. The question is asking regardless of the region or language, so I should look at the totals across all regions and languages.

Looking at the table, I see that it's divided into different regions (1, 3, 4) and languages (French and English). For each region and language combination, there are percentages for male and female workers. 

First, I'll list out the percentages for males and females in each category:

- Agricultural region 1, French-language: Female 35.3%, Male 64.7%
- Agricultural region 1, English-language: Female 28.0%, Male 72.0%
- Agricultural region 3, French-language: Female 41.8%, Male 58.2%
- Agricultural region 3, English-language: Female 30.6%, Male 69.4%
- Agricultural region 4, French-language: Female 35.9%, Male 64.1%
- Agricultural region 4, English-language: Female 26.6%, Male 73.4%

Since the question is about which group has more workers regardless of region or language, and in every case males have a higher percentage, it's clear that overall, there are more male workers in the agricultural sector in 2011.

Answer: male

---

Based on the above example, you need to think step by step to answer the question below.

[Table]

Question:
[QUESTION]

Solution:
'''.strip()




def extract_and_join(input_string):
    matches = re.findall(r'<(.*?)>', input_string)
    
    if matches:
        return ','.join(matches)
    else:
        return input_string


def process_output_to_answer(tex: str, typee: str, table: List[List[str]]) -> List[str]:
    text = deepcopy(tex)
    code = ""
    # print(f"Type: {typee}")
    if typee == "md":
        if "\nAnswer: \n" in text:
            text = text.split("\nAnswer: \n")[1].strip()
        elif "\nAnswer:\n" in text:
            text = text.split("\nAnswer:\n")[1].strip()
        elif "Answer: " in text:
            text = text.split("Answer: ")[1].strip()
        elif "Answer:" in text:
            text = text.split("Answer:")[1].strip()
        text = text.strip("*")
        answer = [text]
    elif typee == "db":
        pred_extracted = extract_sql(text)
        code = fix_answer_sql(pred_extracted, table)
        answer = parse_answer_sql(code, table)
    else:
        if typee == "dict":
            table = list_to_dict(table)
        text = text.strip().strip("\n")
        code = fix_answer(extract_code(text), table, True if typee == "pd" else False)
        answer = parse_answer(code)
        if isinstance(answer, str):
            answer = [answer]
    return answer, code


def evaluate(pred: Union[List[str], str], answer: List[str], file_path: str) -> float:
    if "wikitq" in file_path.lower() or "hitab" in file_path.lower():
        if isinstance(pred, str):
            pred = [pred]
        return evaluate_one_instance(pred, answer)
    elif "tablebench" in file_path.lower():
        if isinstance(pred, list):
            pred = "\n".join(pred)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score("\n".join(answer), pred)
        return scores['rougeL'].fmeasure
    

def extract_subtable(table: List[List[str]], cols: List[str], rows: List[str]):
    new_table = []
    if len(cols) == 0:
        cols = ["*"]
    if len(rows) == 0:
        rows = ["*"]
    if isinstance(cols[0], list):
        cols = cols[0]
    if isinstance(rows[0], list):
        rows = rows[0]
    for ti, t in enumerate(table):
        if str(ti) in rows or "*" in rows:
            new_table.append([t[ci] for ci, c in enumerate(t) if str(ci) in cols or "*" in cols])
    if len(new_table) == 0 or len(new_table[0]) == 0:
        new_table = table
    return new_table

def pack_path(path: str, shot_num: int) -> str:
    return path.format(shot_num=str(shot_num))

prompt_map = {"WikiTQ": PROMPT_ONE, "TableBench": PROMPT_ONE, "HiTab": PROMPT_ONE_HT}

if __name__ == '__main__':
    from utils.evaluate_qa import *
    from utils.table import trans_table, list_to_dict
    from utils.metric_WikiTQ import evaluate_one_instance
    from utils.program import fix_answer, parse_answer, extract_code
    from utils.sql import fix_answer_sql, parse_answer_sql, extract_sql
    from utils.generator import generate_with_llm, consistency

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="llm path")
    parser.add_argument('--lora_name_or_path', type=str, default=None)
    parser.add_argument("--config_file", type=str, help="config path")
    parser.add_argument("--questions_file", type=str, help="questions file")
    parser.add_argument("--shot_num", type=int, nargs='+', help="questions file")
    parser.add_argument('--dump_prompt', action='store_true')
    parser.add_argument("--dump_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    args = parser.parse_args()
    set_seed(args.random_seed)

    model = args.model
    config_file = args.config_file
    questions_file = args.questions_file
    data_size = args.data_size
    if args.lora_name_or_path == 'none':
        args.lora_name_or_path = None
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Data size: {len(data)}")
    if args.data_size and args.data_size < len(data):
        data = random.sample(data, args.data_size)
    if "WikiTQ" in questions_file:
        dataset_name = "WikiTQ"
    elif "TableBench" in questions_file:
        dataset_name = "TableBench"
    elif "HiTab" in questions_file:
        dataset_name = "HiTab"

    for shot_num in args.shot_num:
        for di, d in tqdm(enumerate(data), desc="Processing data"):
            question = d["utterance"]
            d["response"] = ""
            table_format = "md"
            PROMPT = prompt_map[dataset_name]
            dprompt = PROMPT.replace("[QUESTION]", question)
            table = d["table"]
            dprompt = dprompt.replace("[Table]", str(trans_table(table, table_format)))
            d["instruction"] = dprompt

        print(data[-1]["instruction"])
        dump_file = pack_path(args.dump_file, shot_num)

        if args.dump_prompt:
            with open(dump_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        predictions = generate_with_llm(model, data, config, 'chat', lora_name_or_path=args.lora_name_or_path)
        for di, d in enumerate(data):
            record: List[Tuple[str, str, float]] = []
            for pi, p in enumerate(predictions[di]):
                pred, code = process_output_to_answer(p[0], "md", d["table"][0]["table"])
                record.append((p[0], pred, p[1]))
            d["output"], d["pred"] = consistency(record)
            d["output"] = d["output"].split("\n")
            d['correct'] = evaluate(d["pred"], d["answer"], dump_file)
            
        print(data[-1]["output"])
        print(f"#########{dump_file} score: {sum([d['correct'] for d in data]) / len(data)}")

        with open(dump_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

