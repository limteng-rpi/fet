import re
import glob
import json
import torch
import pickle
import random
import constant as C
from allennlp.modules.elmo import batch_to_ids

DIGIT_PATTERN = re.compile('\d')


def mask_to_distance(mask, mask_len, decay=.1):
    start = mask.index(1)
    end = mask_len - list(reversed(mask)).index(1)
    dist = [0] * mask_len
    for i in range(start):
        dist[i] = max(0, 1 - (start - i - 1) * decay)
    for i in range(end, mask_len):
        dist[i] = max(0, 1 - (i - end) * decay)
    return dist


def offset_to_distance(start, end, seq_len, decay=.1):
    dist = [0] * seq_len
    for i in range(start):
        dist[i] = max(0, 1 - (start - i - 1) * decay)
    for i in range(end, seq_len):
        dist[i] = max(0, 1 - (i - end) * decay)
    return dist


class BufferDataset(object):
    def __init__(self,
                 path,
                 buffer_size=100000,
                 max_seq_len=-1,
                 buffer=True,
                 pad=C.PAD_INDEX):
        self.path = path
        self.pad = pad
        self.data = []
        self.files = []
        self.batches = []
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.total_size = 0
        self.buffer = buffer
        self.count_total_size()


    @property
    def chunk_num(self):
        return len(self.chunks)

    @property
    def chunks(self):
        chunks = glob.glob(self.path + '.bin.*')
        return chunks

    def __len__(self):
        return self.total_size

    def count_total_size(self):
        line_idx = 0
        with open(self.path + '.meta', 'r', encoding='utf-8') as r:
            for line_idx, _ in enumerate(r, 1):
                pass
        self.total_size = line_idx

    @staticmethod
    def preprocess(input_file, output_file, label_stoi, chunk_size=10000):
        # shuffle the data
        print('Shuffling and chunking data')
        inst_num = 0
        with open(input_file, 'r', encoding='utf-8') as r:
            for inst_num, _ in enumerate(r, 1):
                pass
        chunk_num = inst_num // chunk_size + (inst_num % chunk_size > 0)
        inst_idxs = [i for i in range(inst_num)]
        random.shuffle(inst_idxs)
        inst_chunk = [0] * inst_num
        for chunk_idx, chunk_start in enumerate(range(0, inst_num, chunk_size)):
            chunk = inst_idxs[chunk_start:chunk_start + chunk_size]
            for inst_idx in chunk:
                inst_chunk[inst_idx] = chunk_idx

        # metadata
        print('Writing metadata')
        with open(output_file + '.meta', 'w', encoding='utf-8') as w:
            for idx, chunk_idx in enumerate(inst_chunk):
                w.write('{}\t{}\n'.format(idx, chunk_idx))

        # split the original file into chunks
        print('Writing chunks')
        writers = [open(output_file + '.txt.{:03d}'.format(idx), 'w')
                   for idx in range(chunk_num)]
        with open(input_file, 'r', encoding='utf-8') as r:
            for inst_idx, line in enumerate(r):
                writers[inst_chunk[inst_idx]].write(line)
        for w in writers:
            w.close()

        # numberize data
        print('Numberizing chunks')
        for chunk_idx in range(chunk_num):
            print('\rProcessed {}/{}'.format(chunk_idx + 1, chunk_num), end='')
            with open(output_file + '.txt.{:03d}'.format(chunk_idx), 'r') as r, \
                open(output_file + '.bin.{:03d}'.format(chunk_idx), 'wb') as w:
                chunk = [BufferDataset.numberize(json.loads(line), label_stoi)
                         for line in r]
                pickle.dump(chunk, w)
        print()

    @staticmethod
    def numberize(inst, label_stoi):
        tokens = inst['tokens']
        tokens = [C.TOK_REPLACEMENT.get(t, t) for t in tokens]
        seq_len = len(tokens)
        char_ids = batch_to_ids([tokens])[0].tolist()
        labels_nbz, men_mask, ctx_mask, men_ids = [], [], [], []
        annotations = inst['annotations']
        anno_num = len(annotations)
        for annotation in annotations:
            mention_id = annotation['mention_id']
            labels = annotation['labels']
            labels = [l.replace('geograpy', 'geography') for l in labels]
            start = annotation['start']
            end = annotation['end']

            men_ids.append(mention_id)
            labels = [label_stoi[l] for l in labels if l in label_stoi]
            labels_nbz.append(labels)
            men_mask.append([1 if i >= start and i < end else 0
                             for i in range(seq_len)])
            ctx_mask.append([1 if i < start or i >= end else 0
                             for i in range(seq_len)])
        return (char_ids, labels_nbz, men_mask, ctx_mask, men_ids, anno_num,
                seq_len)

    def load(self, shuffle=False):
        self.data = []
        if self.buffer:
            while len(self.data) < self.buffer_size and len(
                    self.data) < self.total_size:
                if len(self.files) == 0:
                    self.files = glob.glob(self.path + '.bin.*')
                    if shuffle:
                        random.shuffle(self.files)
                file = self.files.pop()
                if self.max_seq_len > 0:
                    data = pickle.load(open(file, 'rb'))
                    for inst in data:
                        if inst[-1] < self.max_seq_len:
                            self.data.append(inst)
                else:
                    self.data.extend(pickle.load(open(file, 'rb')))
        else:
            files = glob.glob(self.path + '.bin.*')
            if shuffle:
                random.shuffle(files)
            # Load all files
            for file in files:
                if self.max_seq_len > 0:
                    data = pickle.load(open(file, 'rb'))
                    for inst in data:
                        if inst[-1] < self.max_seq_len:
                            self.data.append(inst)
                else:
                    self.data.extend(pickle.load(open(file, 'rb')))

    def sample_batches(self, batch_size=100, drop_last=False,
                       shuffle=False):
        # Load data
        if self.buffer or len(self.data) == 0:
            self.load(shuffle)
        inst_num = len(self.data)
        inst_idxs = [i for i in range(inst_num)]
        # Shuffle instance indexes
        if shuffle:
            random.shuffle(inst_idxs)
        # Split the indexes into batches
        self.batches = [inst_idxs[i:i + batch_size]
                        for i in range(0, inst_num, batch_size)]
        # Drop the last batch
        if drop_last and len(self.batches[-1]) != batch_size:
            self.batches = self.batches[:-1]

    def next_batch(self, label_size, batch_size=100, drop_last=False,
                   shuffle=False, gpu=True):
        # Load data and generate batches
        if len(self.batches) == 0:
            self.sample_batches(batch_size, drop_last, shuffle)

        # Get the batch
        batch_idxs = self.batches.pop()
        batch = [self.data[idx] for idx in batch_idxs]
        # Process the batch
        seq_lens = [x[-1] for x in batch]
        max_seq_len = max(seq_lens)

        batch_char_ids = []
        batch_labels = []
        batch_men_mask = []
        batch_dist = []
        batch_ctx_mask = []
        batch_gathers = []
        batch_men_ids = []
        for inst_idx, inst in enumerate(batch):
            char_ids, labels, men_mask, ctx_mask, men_ids, anno_num, seq_len = inst
            batch_char_ids.append(char_ids + [[self.pad] * C.ELMO_MAX_CHAR_LEN
                                              for _ in range(max_seq_len - seq_len)])
            for ls in labels:
                batch_labels.append([1 if l in ls else 0
                                     for l in range(label_size)])
            for mask in men_mask:
                batch_men_mask.append(mask + [self.pad] * (max_seq_len - seq_len))
                batch_dist.append(mask_to_distance(mask, seq_len)
                                  + [self.pad] * (max_seq_len - seq_len))
            for mask in ctx_mask:
                batch_ctx_mask.append(mask + [self.pad] * (max_seq_len - seq_len))
            batch_gathers.extend([inst_idx] * anno_num)
            batch_men_ids.extend(men_ids)

        if gpu:
            batch_char_ids = torch.cuda.LongTensor(batch_char_ids)
            batch_labels = torch.cuda.FloatTensor(batch_labels)
            batch_men_mask = torch.cuda.FloatTensor(batch_men_mask)
            batch_ctx_mask = torch.cuda.FloatTensor(batch_ctx_mask)
            batch_gathers = torch.cuda.LongTensor(batch_gathers)
            batch_dist = torch.cuda.FloatTensor(batch_dist)

        else:
            batch_char_ids = torch.LongTensor(batch_char_ids)
            batch_labels = torch.FloatTensor(batch_labels)
            batch_men_mask = torch.FloatFloatTensorTensor(batch_men_mask)
            batch_ctx_mask = torch.LongTensor(batch_ctx_mask)
            batch_gathers = torch.LongTensor(batch_gathers)
            batch_dist = torch.FloatTensor(batch_dist)

        return (batch_char_ids, batch_labels, batch_men_mask, batch_ctx_mask,
                batch_dist, batch_gathers, batch_men_ids)

    def all_batches(self, label_size, batch_size, drop_last=False,
                    shuffle=False, gpu=True):
        self.sample_batches(batch_size, drop_last, shuffle)
        while len(self.batches) > 0:
            yield self.next_batch(label_size, batch_size, drop_last,
                                  shuffle, gpu)
