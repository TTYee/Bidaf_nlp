# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
BiDAF model
"""

import argparse
import json
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.dataset as ds
import mindspore.context as context

# from model import BiDAF
from model_nlp import Encoder, Head, BiDAF
from data import load_vocab
from evaluate import evaluate
from ema import EMA

# mindspore.set_context(mode=context.PYNATIVE_MODE ,max_call_depth=10000)
# mindspore.set_context(mode=context.GRAPH_MODE ,max_call_depth=10000, enable_graph_kernel=True)
mindspore.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)


def train_loop(model, dataset, loss_fn, optimizer, ema):
    # Define forward function
    def forward_fn(c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx):
        logits = model(c_char, q_char, c_word, q_word, c_lens, q_lens)
        loss = loss_fn(logits[0], s_idx) + loss_fn(logits[1], e_idx)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx):
        (loss, _), grads = grad_fn(c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx, _) \
        in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx)
        for name, param in model.parameters_and_names():
            if param.requires_grad:
                ema.update(name, param.data)
                param.set_data(ema.get(name))

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_loop(model, dataset, vocab, loss_fn):
    model.set_train(False)
    loss = 0
    answers = dict()

    for c_char, q_char, c_word, q_word, c_lens, q_lens, s_idx, e_idx, ids in dataset.create_tuple_iterator():
        p1, p2 = model(c_char, q_char, c_word, q_word, c_lens, q_lens)
        batch_loss = loss_fn(p1, s_idx) + loss_fn(p2, e_idx)
        loss += batch_loss

        # [batch, c_len, c_len]
        batch_size, c_len = p1.shape
        ls = nn.LogSoftmax(axis=1)
        mask = mnp.tril((ops.ones((c_len, c_len), type=mindspore.float32) * float('-inf')),
                         k=-1).expand_dims(0).broadcast_to((batch_size, -1, -1))
        mask = mnp.where(ops.isnan(mask), ops.zeros_like(mask), mask)
        score = (ls(p1).expand_dims(2) + ls(p2).expand_dims(1)) + mask
        s_idx, score = ops.max(score, axis=1)
        e_idx, score = ops.max(score, axis=1)
        s_idx = ops.gather_elements(s_idx, 1, e_idx.view(-1, 1)).squeeze(axis=1)

        for i in range(batch_size):
            answer_id = ids.asnumpy()[i]
            answer = c_word[i][s_idx[i].asnumpy().item():e_idx[i].asnumpy().item()+1]

            answer_list = []
            for idx in answer:
                idx = idx.asnumpy().item()
                if idx < 0:
                    idx = idx + 188744
                answer_list.append(vocab.ids_to_tokens(idx))
            answer = ' '.join(answer_list)
            answers[answer_id.item()] = answer

    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        squad_data = dataset_json['data']
    exact_match, f1 = evaluate(squad_data, answers)
    print(f"Test: \n EM: {exact_match:.3f}, F1: {f1:.3f}, Avg loss: {loss.asnumpy().item():>8f} \n")


parser = argparse.ArgumentParser()
parser.add_argument('--char_dim', default=8, type=int)
parser.add_argument('--char_channel_width', default=5, type=int)
parser.add_argument('--char_channel_size', default=100, type=int)
parser.add_argument('--context_threshold', default=400, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--exp_decay_rate', default=0.999, type=float)
parser.add_argument('--hidden_size', default=100, type=int)
parser.add_argument('--print_freq', default=250, type=int)
parser.add_argument('--word_dim', default=100, type=int)

parser.add_argument('--train_batch_size', default=60, type=int)
parser.add_argument('--dev_batch_size', default=100, type=int)
parser.add_argument('--epoch', default=6, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--learning_rate', default=0.5, type=float)

parser.add_argument('--train_file', default='train-v1.1.json')
parser.add_argument('--dev_file', default='dev-v1.1.json')
parser.add_argument('--dataset_file', default='.data/dev-v1.1.json')
args = parser.parse_args()

# load datasets
char_vocab, word_vocab = load_vocab()
embeddings = np.load(".data/embeddings.npy")

train_file = '/run/train.mindrecord'
valid_file = '/run/dev.mindrecord'

squad_train = ds.MindDataset(train_file, shuffle=True,
                             columns_list=["c_char", "q_char", "c_word", "q_word",
                                           "c_lens", "q_lens", "s_idx", "e_idx", "ids"])
squad_valid = ds.MindDataset(valid_file, shuffle=False,
                             columns_list=["c_char", "q_char", "c_word", "q_word",
                                           "c_lens", "q_lens", "s_idx", "e_idx", "ids"])

squad_train = squad_train.batch(args.train_batch_size)
squad_valid = squad_valid.batch(args.dev_batch_size)

# define Models & Loss & Optimizer
char_vocab_size = len(char_vocab.vocab())
char_dim = args.char_dim
char_channel_width = args.char_channel_width
char_channel_size = args.char_channel_size
pad_idx = word_vocab.tokens_to_ids('<pad>')
hidden_size = args.hidden_size
dropout = args.dropout
lr = args.learning_rate
epoch = args.epoch

# net = BiDAF(char_vocab_size, char_dim, char_channel_width, char_channel_size,
#             embeddings, pad_idx, hidden_size, dropout)

encoder = Encoder(char_vocab_size, char_vocab, char_dim, char_channel_size, char_channel_width, word_vocab,
                  embeddings, pad_idx, hidden_size, dropout)
head = Head(hidden_size, dropout)
net = BiDAF(encoder, head)

ema = EMA(args.exp_decay_rate)
for name, param in net.parameters_and_names():
    if param.requires_grad:
        ema.register(name, param.data)

loss = nn.CrossEntropyLoss()
optimizer = nn.Adadelta(net.trainable_params(), learning_rate=lr)

for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(net, squad_train, loss, optimizer, ema)
    test_loop(net, squad_valid, word_vocab, loss)
print("Done!")
