# Requirment
- Python 3.5+
- PyTorch 1.0+
- tqdm
- pytorch_pretrained_bert (transformers is recommended)
- allennlp

# File Format
We convert datasets in our experiments in two steps:
1. We convert them to a JSON-based format as follows (in the actual file, each line is a JSON object without linebreak and indentation).
```
{
    "tokens": ["\"", "My", "direction", "to", "city", "staff", "and", "our", "experts", "is", "to", "focus", "on", "two", "goals", ":", "1", ")", "Seriously", "explore", "and", "consider", "the", "opportunity", ",", "and", "2", ")", "Ensure", "that", "taxpayers", "and", "the", "city", "of", "Seattle", "are", "protected", ",", "particularly", "in", "light", "of", "the", "public", "'s", "direction", "through", "I-91", ",", "\"", "McGinn", "said", "in", "an", "emailed", "statement", "."],
    "annotations": [
        {"mention_id": "65-0", "mention": "Seattle", "start": 35, "end": 36, "labels": ["/location/city", "/location"]},
        {"mention_id": "65-1", "mention": "I-91", "start": 48, "end": 49, "labels": ["/written_work"]},
        {"mention_id": "65-2", "mention": "McGinn", "start": 51, "end": 52, "labels": ["/person"]}
    ]
}
```
2. Because fine-grained entity typing training sets are usually large, we shuffle the training examples and split them into chunks so that we don't need to load the whole training set into the memory. The conversion can be done using `data.BufferDataset.preprocess()`.

Parameters:
- `input_file`: A JSON file formatted as step 1 shows.
- `output_file`: Path to the output files. Note that because the input file will be split into many files, this path is actually the prefix of all output files. For example, if `output_file=/output/path/train`, the script will generate `train.meta`, `train.txt.000` (JSON format), `train.txt.001`, ..., `train.txt.129`, and `train.bin.000` (binary, converted from `train.txt.000`), `train.bin.001`, ..., `train.bin.129` in the directory `/output/path/`.
- `label_stoi`: A label to index dict object.
- `chunk_size`: The number of sentences in each chunk (default=10000).

# Training

Suppose all required files are located in `/path/to/data`, and the converted data sets are saved in `/path/to/data/buffer`, run the following command to train the model:
```
python train.py --train /path/to/data/buffer/train --dev /path/to/data/buffer/dev 
--test /path/to/data/buffer/test --ontology /path/to/data/type.txt --output /path/to/output
--elmo_option /path/to/data/eng.original.5.5b.json --elmo_weight /path/to/data/eng.original.5.5b.hdf5
--batch_size 200 --max_epoch 100 --eval_step 200 --gpu
```

* `ontology` is a text file that lists all type labels (one per line).
* Pre-trained ELMo embeddings can be downloaded from https://allennlp.org/elmo

# Reference

- Lin, Ying, and Heng Ji. "An Attentive Fine-Grained Entity Typing Model with Latent Type Representation." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019. \[[pdf](https://www.aclweb.org/anthology/D19-1641.pdf)\]

```
@inproceedings{lin2019fet,
    title = {An Attentive Fine-Grained Entity Typing Model with Latent Type Representation},
    author = {Lin, Ying and Ji, Heng},
    booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    year = {2019},
}
```
