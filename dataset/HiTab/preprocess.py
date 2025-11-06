import sys
import json

from typing import List, Dict, Any

sys.path.append('.')


def string_to_list(table: str) -> List[List[str]]:
    table = table.split('[TAB] ')[-1]
    # if "| | " in table:
    #     print(table.replace("| | ", " | - | "))
    rows = table.replace("| | ", " | - | ").split(' [SEP] ')
    results = []
    for row in rows:
        row = row.strip('|').strip()
        results.append(row.split(' | '))
    return results


if __name__ == '__main__':
    from utils.table import list_to_dict

    for part in ['train', 'test']:
        with open(f'./dataset/HiTab/raw/{part}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = []
        for idx, d in enumerate(data):
            if 'input' not in d:
                d['input'] = d['input_seg']
            results.append({
                'id': f'HiTab-{part}-{idx}',
                'table': [{
                    "caption": d['input'].split('[TLE] The table caption is ')[-1].split('. [TAB]')[0],
                    # "table": list_to_dict(string_to_list(d['input']))
                    "table": string_to_list(d['input'])
                }],
                'utterance': d['question'],
                'answer': [d['output']]
            })
        with open(f'./dataset/HiTab/{part}.list.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
