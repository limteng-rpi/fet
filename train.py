import os
import time
import tqdm
import torch

from argparse import ArgumentParser
from collections import defaultdict
from pytorch_pretrained_bert import BertAdam
from model import HFet
from data import BufferDataset
from scorer import calculate_metrics
from util import load_ontology, save_result


arg_parser = ArgumentParser()
arg_parser.add_argument('--train')
arg_parser.add_argument('--dev')
arg_parser.add_argument('--test')
arg_parser.add_argument('--svd')
arg_parser.add_argument('--ontology')
arg_parser.add_argument('--output')
arg_parser.add_argument('--lr', type=float, default=5e-5)
arg_parser.add_argument('--max_epoch', type=int, default=5)
arg_parser.add_argument('--batch_size', type=int, default=200)
arg_parser.add_argument('--elmo_option')
arg_parser.add_argument('--elmo_weight')
arg_parser.add_argument('--elmo_dropout', type=float, default=.5)
arg_parser.add_argument('--repr_dropout', type=float, default=.2)
arg_parser.add_argument('--dist_dropout', type=float, default=.2)
arg_parser.add_argument('--gpu', action='store_true')
arg_parser.add_argument('--device', type=int, default=0)
arg_parser.add_argument('--weight_decay', type=float, default=0.001)
arg_parser.add_argument('--latent_size', type=int, default=0)
arg_parser.add_argument('--buffer_size', type=int, default=100000)
arg_parser.add_argument('--eval_step', type=int, default=1000)
args = arg_parser.parse_args()

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

# Output directory
output_dir = os.path.join(args.output, timestamp)
model_mac = os.path.join(output_dir, 'best_mac.mdl')
model_mic = os.path.join(output_dir, 'best_mic.mdl')
result_dev = os.path.join(output_dir, 'dev.best.tsv')
result_test = os.path.join(output_dir, 'test.best.tsv')
os.mkdir(output_dir)

# Set GPU device
gpu = torch.cuda.is_available() and args.gpu
if gpu:
    torch.cuda.set_device(args.device)

# Load data sets
print('Loading data sets')
train_set = BufferDataset(args.train, buffer_size=args.buffer_size)
dev_set = BufferDataset(args.dev, buffer_size=args.buffer_size)
test_set = BufferDataset(args.test, buffer_size=args.buffer_size)

# Vocabulary
label_stoi = load_ontology(args.ontology)
label_itos = {i: s for s, i in label_stoi.items()}
label_size = len(label_stoi)
print('Label size: {}'.format(len(label_stoi)))

# Build model
model = HFet(label_size,
             elmo_option=args.elmo_option,
             elmo_weight=args.elmo_weight,
             elmo_dropout=args.elmo_dropout,
             repr_dropout=args.repr_dropout,
             dist_dropout=args.dist_dropout,
             latent_size=args.latent_size,
             svd=args.svd
             )
if gpu:
    model.cuda()

batch_size = args.batch_size
batch_num = len(train_set) // batch_size
total_step = args.max_epoch * batch_num
optimizer = BertAdam(filter(lambda x: x.requires_grad, model.parameters()),
                         lr=args.lr, warmup=.1,
                         weight_decay=args.weight_decay,
                         t_total=total_step)
state = {
    'model': model.state_dict(),
    'args': vars(args),
    'vocab': {'label': label_stoi}
}
global_step = 0
best_scores = {
        'best_acc_dev': 0, 'best_mac_dev': 0, 'best_mic_dev': 0,
        'best_acc_test': 0, 'best_mac_test': 0, 'best_mic_test': 0
    }
for epoch in range(args.max_epoch):
    print('-' * 20, 'Epoch {}'.format(epoch), '-' * 20)
    start_time = time.time()

    epoch_loss = []
    progress = tqdm.tqdm(total=batch_num, mininterval=1,
                         desc='Epoch: {}'.format(epoch))
    for batch_idx in range(batch_num):
        global_step += 1
        progress.update(1)
        optimizer.zero_grad()
        batch = train_set.next_batch(label_size, batch_size,
                                     drop_last=True, shuffle=True,
                                     gpu=gpu)
        (
            elmos, labels, men_masks, ctx_masks, dists,
            gathers, men_ids,
        ) = batch
        loss = model.forward(elmos, labels, men_masks, ctx_masks, dists,
                             gathers, None)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        if global_step % args.eval_step != 0 and global_step != total_step:
            continue

        # Dev set
        best_acc, best_mac, best_mic = False, False, False
        results = defaultdict(list)
        for batch in dev_set.all_batches(label_size, batch_size=100, gpu=gpu):
            elmo_ids, labels, men_masks, ctx_masks, dists, gathers, men_ids = batch
            preds = model.predict(elmo_ids, men_masks, ctx_masks, dists, gathers)
            results['gold'].extend(labels.int().data.tolist())
            results['pred'].extend(preds.int().data.tolist())
            results['ids'].extend(men_ids)
        metrics = calculate_metrics(results['gold'], results['pred'])
        print('---------- Dev set ----------')
        print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
        print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.macro_prec,
            metrics.macro_rec,
            metrics.macro_fscore))
        print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.micro_prec,
            metrics.micro_rec,
            metrics.micro_fscore))
        # Save model
        if metrics.accuracy > best_scores['best_acc_dev']:
            best_acc = True
            best_scores['best_acc_dev'] = metrics.accuracy
        if metrics.macro_fscore > best_scores['best_mac_dev']:
            best_mac = True
            best_scores['best_mac_dev'] = metrics.macro_fscore
            print('Saving new best macro F1 model')
            torch.save(state, model_mac)
            save_result(results, label_itos, result_dev)
        if metrics.micro_fscore > best_scores['best_mic_dev']:
            best_mic = True
            best_scores['best_mic_dev'] = metrics.micro_fscore
            print('Saving new best micro F1 model')
            torch.save(state, model_mic)

        # Test set
        results = defaultdict(list)
        for batch in test_set.all_batches(label_size, batch_size=100, gpu=gpu):
            elmo_ids, labels, men_masks, ctx_masks, dists, gathers, men_ids = batch
            preds = model.predict(elmo_ids, men_masks, ctx_masks, dists,
                                  gathers)
            results['gold'].extend(labels.int().data.tolist())
            results['pred'].extend(preds.int().data.tolist())
            results['ids'].extend(men_ids)
        metrics = calculate_metrics(results['gold'], results['pred'])
        print('---------- Test set ----------')
        print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
        print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.macro_prec,
            metrics.macro_rec,
            metrics.macro_fscore))
        print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.micro_prec,
            metrics.micro_rec,
            metrics.micro_fscore))
        if best_acc:
            best_scores['best_acc_test'] = metrics.accuracy
        if best_mac:
            best_scores['best_mac_test'] = metrics.macro_fscore
            save_result(results, label_itos, result_test)
        if best_mic:
            best_scores['best_mic_test'] = metrics.micro_fscore

        for k, v in best_scores.items():
            print('{}: {:.2f}'.format(k.replace('_', ' '), v))
    progress.close()



