import rouge
import numpy as np

from typing import Union, List, Dict, Any
from sklearn.metrics import f1_score

# class EvaluationMetrics:
#     def __init__(self):
#         self.rouge = rouge.Rouge()

#     def compute_em(self, pred_answer, gold_answer):
#         if isinstance(pred_answer, dict):
#             pred_answer = pred_answer['pred_answer']
#         if isinstance(gold_answer, dict):
#             gold_answer = gold_answer['answer']
        
#         return int(pred_answer == gold_answer)

#     def compute_rouge(self, pred_answer, gold_answer):
#         if isinstance(pred_answer, dict):
#             pred_answer = pred_answer['pred_answer']
#         if isinstance(gold_answer, dict):
#             gold_answer = gold_answer['answer']

#         scores = self.rouge.get_scores(pred_answer, gold_answer)
#         return scores[0]['rouge-l']['f']

#     def evaluate(self, pred_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]], 
#                  gold_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, float]:
#         em_scores = []
#         rouge_scores = []

#         # 处理单条数据和列表数据
#         if isinstance(pred_answer, (list, dict)) and isinstance(gold_answer, (list, dict)):
#             pred_list = pred_answer if isinstance(pred_answer, list) else [pred_answer]
#             gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]

#             for pred, gold in zip(pred_list, gold_list):
#                 em_scores.append(self.compute_em(pred, gold))
#                 rouge_scores.append(self.compute_rouge(pred, gold))
        
#         # 计算平均值
#         avg_em = np.mean(em_scores) if em_scores else 0.0
#         avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0

#         return {
#             "average_em": avg_em,
#             "average_rouge_l": avg_rouge
#         }
    


class EvaluationMetrics:
    def __init__(self):
        pass

    @staticmethod
    def calculate_em(pred_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]], 
                     gold_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> float:
        def compute_exact_match(pred, gold):
            return 1 if pred == gold else 0

        if isinstance(pred_answer, list) and isinstance(gold_answer, list):
            em_scores = [compute_exact_match(pred['pred_answer'], gold['answer']) 
                         for pred, gold in zip(pred_answer, gold_answer)]
        elif isinstance(pred_answer, dict) and isinstance(gold_answer, dict):
            em_scores = [compute_exact_match(pred_answer['pred_answer'], gold_answer['answer'])]
        else:
            em_scores = [compute_exact_match(pred_answer, gold_answer)]

        return sum(em_scores) / len(em_scores)

    @staticmethod
    def calculate_f1(pred_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]], 
                     gold_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> float:
        def compute_f1(pred, gold):
            pred_set = set(pred) if isinstance(pred, list) else {pred}
            gold_set = set(gold) if isinstance(gold, list) else {gold}
            if not pred_set or not gold_set:
                return 0.0
            intersection = pred_set.intersection(gold_set)
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(gold_set) if gold_set else 0
            return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        if isinstance(pred_answer, list) and isinstance(gold_answer, list):
            f1_scores = [compute_f1(pred['pred_answer'], gold['answer']) 
                          for pred, gold in zip(pred_answer, gold_answer)]
        elif isinstance(pred_answer, dict) and isinstance(gold_answer, dict):
            f1_scores = [compute_f1(pred_answer['pred_answer'], gold_answer['answer'])]
        else:
            f1_scores = [compute_f1(pred_answer, gold_answer)]

        return sum(f1_scores) / len(f1_scores)
    
    @staticmethod
    def evaluate(pred_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]], 
                 gold_answer: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, float]:
        em_score = EvaluationMetrics.calculate_em(pred_answer, gold_answer)
        f1_score = EvaluationMetrics.calculate_f1(pred_answer, gold_answer)
        return {
            'EM': em_score,
            'F1': f1_score
        }

# Example usage




if __name__ == '__main__':
    # evaluator = EvaluationMetrics()

    # # 示例数据
    pred = [
        {"pred_answer": "答案1"},
        {"pred_answer": "答案2"}
    ]
    gold = [
        {"answer": "答案1"},
        {"answer": "错误答案"}
    ]

    # results = evaluator.evaluate(pred, gold)
    # print(results)  # 输出: {'average_em': 0.5, 'average_rouge_l': 0.75}
    evaluator = EvaluationMetrics()
    score = evaluator.evaluate(pred, gold)
    print(score)

