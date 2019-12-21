# File Format
We convert datasets in our experiments in two steps:
1. We convert them to a JSON-based format as follows (each line is a sentence).
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
2. Because fine-grained entity typing training sets are usually large, we shuffle the training examples and split them into chunks of size 10,000 using `BufferDataset.preprocess()`.

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
