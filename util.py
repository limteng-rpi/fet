import json


def generate_ontology_file(input_files, output_file):
    labels = set()
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
                for anno in inst['annotations']:
                    labels.update(anno['labels'])
    print(labels)
    json.dump([{'labels': list(labels)}],
              open(output_file, 'w', encoding='utf-8'))
    
    
def generate_ontology_file_txt(input_files, output_file):
    labels = set()
    for input_file in input_files:
        with open(input_file) as r:
            for line in r:
                inst = json.loads(line)
                for anno in inst['annotations']:
                    labels.update(anno['labels'])
    print(labels)
    with open(output_file, 'w') as w:
        for label in labels:
            w.write(label + '\n')


def load_ontology(path):
    """
    :param path:
    :return:
    """
    label_stoi = {}
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            label_stoi[line.strip()] = len(label_stoi)
    return label_stoi


def counter_to_vocab(counter, offset=0, pads=None, min_count=0):
    """Convert a counter to a vocabulary.
    :param count: A counter to convert.
    :param offset: Begin start offset.
    :param pads: A list of padding (str, index) pairs.
    :param min_count: Minimum count.
    :param ignore_case: Ignore case.
    :return: Vocab dict.
    """
    vocab = {}
    for token, freq in counter.items():
        if freq > min_count:
            vocab[token] = len(vocab) + offset
    if pads:
        for k, v in pads:
            vocab[k] = v

    return vocab


def save_result(results, label_itos, path):
    with open(path, 'w', encoding='utf-8') as w:
        for gold, pred, men_id in zip(results['gold'],
                                      results['pred'],
                                      results['ids']):
            gold_labels = [label_itos[i] for i, l in enumerate(gold) if l]
            pred_labels = [label_itos[i] for i, l in enumerate(pred) if l]
            gold_labels.sort()
            pred_labels.sort()
            w.write('{}\t{}\t{}\n'.format(men_id,
                                          ','.join(gold_labels),
                                          ','.join(pred_labels)))
