from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from difflib import SequenceMatcher

def compute_bleu_from_ids(predictions, references):
    """
    使用 str 计算 BLEU 分数。
    Args:
        predictions (List[str]): 模型预测的文本
        references (List[str]): 参考答案的文本

    Returns:
        float: BLEU 分数。
    """
    # 确保参考序列格式符合 corpus_bleu 的要求
    predictions = [pred.split() for pred in predictions]
    references = [[ref.split()] for ref in references]
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smooth)
    return bleu_score


def compute_rouge_from_ids(predictions, references):
    """
    使用文本计算 ROUGE 分数。
    Args:
        predictions (List[str]): 模型预测的文本。
        references (List[str]): 参考答案的文本。

    Returns:
        Dict[str, float]: 包含 ROUGE-1、ROUGE-2 和 ROUGE-L 的分数。
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    count = len(predictions)

    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        rouge_scores["rouge1"] += score["rouge1"].fmeasure
        rouge_scores["rouge2"] += score["rouge2"].fmeasure
        rouge_scores["rougeL"] += score["rougeL"].fmeasure

    # 平均分
    return {k: v / count for k, v in rouge_scores.items()}


def open_question_metrics(predictions, references,special_ids = [151643]):
    """
    计算开放式问题的 BLEU 和 ROUGE 分数。
    Args:
        predictions (List[str]): 模型预测的文本。
        references (List[str]): 参考答案的文本。
        special_ids (int): 用于填充的索引。

    Returns:
        Dict[str, float]: 包含 BLEU 和 ROUGE 的分数。
    """
    # 移除 padding 
    decoded_predictions = []
    decoded_labels = []

    for pred, label in zip(predictions, references):
        pred = [token for token in pred if token not in  special_ids]
        label = [token for token in label if token not in special_ids]
        decoded_predictions.append(pred)
        decoded_labels.append(label)

    # 计算 BLEU
    bleu_score = compute_bleu_from_ids(predictions, references)

    # 计算 ROUGE
    rouge_scores = compute_rouge_from_ids(predictions, references)

    return {"BLEU": bleu_score, **rouge_scores}

def compute_rul(predictions, references):
    """
    计算 RUL 分数。

    Args:
        predictions (List[str]): 模型预测的数值。
        references (List[str]): 参考答案的数值。

    Returns:
        Dict[str, float]: 包含 MAE 和 RMSE 的分数。
    """
    # 将字符串转换为数值
    predictions = [float(pred) if pred.replace('.', '', 1).isdigit() else 30 for pred in predictions]
    references = [float(ref) for ref in references]

    # 计算 MAE (Mean Absolute Error)
    mae = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)

    # 计算 RMSE (Root Mean Squared Error)
    mse = sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)
    rmse = mse ** 0.5

    return {"MAE": mae, "RMSE": rmse,"MSE":mse}




def closed_question_metrics(predictions, references,special_id=[151643]):
    """
    计算不定项选择题的评估指标：精确率、召回率、F1 分数和完全匹配率。

    Args:
        predictions (List[str]): 模型预测的答案列表，单选或多选用空格分隔（如 'a b e'）。
        references (List[str]): 正确答案的列表，单选或多选用空格分隔（如 'a b'）。

    Returns:
        dict: 包含精确率、召回率、F1 和完全匹配率的结果。
    """
    tp, fp, fn = 0, 0, 0
    exact_match_count = 0

    for pred, ref in zip(predictions, references):
        # 将字符串转换为集合
        pred_set = set(pred.split())
        ref_set = set(ref.split())

        #将pred_set中的字符小写
        pred_set = {token.lower() for token in pred_set}
        # 去除pred_set中非选项的字符(从a到z)

        pred_set = {token for token in pred_set if token in['a','b','c','d','e','f','g','h'
                                                            ,'i','j','k','l','m','n','o','p','q',
                                                            'r','s','t','u','v','w','x','y','z']}

        # 计算 True Positives, False Positives, False Negatives
        tp += len(pred_set & ref_set)  # 正确预测的选项
        fp += len(pred_set - ref_set)  # 错误预测的选项
        fn += len(ref_set - pred_set)  # 漏掉的正确选项

        # 完全匹配检查
        if pred_set == ref_set:
            exact_match_count += 1

    # 计算指标
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    exact_match_accuracy = exact_match_count / len(references) if len(references) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_accuracy": exact_match_accuracy,
    }

# # 示例数据
# predictions = ['a', 'a token', 'a', 'a', 'b', 'b', 'a b e', 'b', 'a', 'a', 'a', 'b']
# references = ['a', 'a', 'a', 'c', 'b', 'b', 'a b', 'b', 'a', 'a', 'a', 'b']

# # 调用函数
# metrics = closed_question_metrics(predictions, references)
# print(metrics)
def search_position(valid_positions, context_size=2):
    """
    在每个元素的前后各添加 context_size 个元素。

    Args:
        valid_positions (List[List[int]]): 原始位置列表。
        context_size (int): 前后添加的元素数量。

    Returns:
        List[List[int]]: 添加了前后元素的新位置列表。
    """
    new_positions = []
    for positions in valid_positions:
        new_pos = set()  # 使用集合来避免重复元素
        for pos in positions:
            # 添加前面的元素
            new_pos.update(range(max(0, pos - context_size), pos))
            # 添加当前元素
            new_pos.add(pos)
            # 添加后面的元素
            new_pos.update(range(pos + 1, pos + context_size + 1))
        new_positions.append(sorted(new_pos))  # 转换回列表并排序
    return new_positions

