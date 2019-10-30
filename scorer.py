from collections import namedtuple

TYPING_METRICS = namedtuple('TypingMetrics',
                            ['accuracy',
                             'micro_prec', 'micro_rec', 'micro_fscore',
                             'macro_prec', 'macro_rec', 'macro_fscore',
                             ])

def calculate_fscore(recall, precision):
    fscore = 0 if precision + recall == 0 else \
        2.0 * (precision * recall) / (precision + recall)
    return fscore

def calculate_strict_accuracy(golds, preds):
    total_num = len(golds)
    correct_num = 0
    for gold, pred in zip(golds, preds):
        if gold == pred:
            correct_num += 1
    return correct_num / total_num * 100.0

def calculate_strict_accuracy_sparse(golds, preds):
    total_num = len(golds)
    correct_num = 0
    for gold, pred in zip(golds, preds):
        gold = sorted(gold)
        pred = sorted(pred)
        if gold == pred:
            correct_num += 1
    return correct_num / total_num * 100.0

def calculate_macro_fscore(golds, preds):
    total_gold_num, total_pred_num = 0, 0
    precision, recall = 0, 0
    for gold, pred in zip(golds, preds):
        gold_num = sum(gold)
        pred_num = sum(pred)
        total_gold_num += (1 if gold_num > 0 else 0)
        total_pred_num += (1 if pred_num > 0 else 0)
        overlap = sum([i and j for i, j in zip(gold, pred)])
        precision += (0 if pred_num == 0 else overlap / pred_num)
        recall += (0 if gold_num == 0 else overlap / gold_num)
    precision = precision / total_pred_num if total_pred_num else 0
    recall = recall / total_gold_num if total_gold_num else 0
    fscore = 0 if precision + recall == 0 else \
        2.0 * (precision * recall) / (precision + recall)
    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_macro_fscore_sparse(golds, preds):
    total_gold_num, total_pred_num = 0, 0
    precision, recall = 0, 0
    for gold, pred in zip(golds, preds):
        gold_num, pred_num = len(gold), len(pred)
        total_gold_num += (1 if gold_num > 0 else 0)
        total_pred_num += (1 if pred_num > 0 else 0)
        overlap = len(set(gold).intersection(set(pred)))
        precision += (0 if pred_num == 0 else overlap / pred_num)
        recall += (0 if gold_num == 0 else overlap / gold_num)
    precision = precision / total_pred_num if total_pred_num else 0
    recall = recall / total_gold_num if total_gold_num else 0
    fscore = calculate_fscore(recall, precision)
    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_micro_fscore(golds, preds):
    overlap, gold_total, pred_total = 0, 0, 0
    for gold, pred in zip(golds, preds):
        gold_total += sum(gold)
        pred_total += sum(pred)
        overlap += sum([i and j for i, j in zip(gold, pred)])
    precision = 0 if pred_total == 0 else overlap / pred_total
    recall = 0 if gold_total == 0 else overlap / gold_total
    fscore = 0 if precision + recall == 0 else \
        2.0 * (precision * recall) / (precision + recall)
    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_micro_fscore_sparse(golds, preds):
    overlap, gold_total, pred_total = 0, 0, 0
    for gold, pred  in zip(golds, preds):
        gold_total += len(gold)
        pred_total += len(pred)
        overlap += len(set(gold).intersection(set(pred)))
    precision = 0 if pred_total == 0 else overlap / pred_total
    recall = 0 if gold_total == 0 else overlap / gold_total
    fscore = calculate_fscore(recall, precision)
    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_metrics(golds, preds):
    accuracy = calculate_strict_accuracy(golds, preds)
    micro_prec, micro_rec, micro_fscore = calculate_micro_fscore(golds, preds)
    macro_prec, macro_rec, macro_fscore = calculate_macro_fscore(golds, preds)
    return TYPING_METRICS(
        accuracy=accuracy,
        micro_prec=micro_prec, micro_rec=micro_rec, micro_fscore=micro_fscore,
        macro_prec=macro_prec, macro_rec=macro_rec, macro_fscore=macro_fscore,
    )