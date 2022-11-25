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


import json
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.context as context
from model_nlp import Encoder, Head, BiDAF
from mindnlp.engine.trainer import Trainer
from data import load_vocab
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.dataset.register import load, process

# mindspore.set_context(mode=context.PYNATIVE_MODE ,max_call_depth=10000)
# mindspore.set_context(mode=context.GRAPH_MODE ,max_call_depth=10000, enable_graph_kernel=True)
mindspore.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)


# load datasets
squad_train, squad_dev = load('squad1', shuffle=False, proxies={"https": "http://172.20.106.122:7890"})
print(squad_train.get_col_names())

# load vocab
with open('.data/char_vocab.json', mode='r', encoding='utf-8') as json_file:
    char_dict = json.load(json_file)
    char_vocab = ds.text.Vocab.from_dict(char_dict)
with open('.data/word_vocab.json', mode='r', encoding='utf-8') as json_file:
    word_dict = json.load(json_file)
    word_vocab = ds.text.Vocab.from_dict(word_dict)

# load word embedding
word_embeddings = np.load(".data/embeddings.npy")
tokenizer = BasicTokenizer(True)

# process dataset
print("============================Processing dataset, please wait a minute=============================")
squad_train = process('squad1', squad_train, word_vocab, char_vocab, tokenizer=tokenizer,\
                   max_context_len=768, max_question_len=64, max_char_len=48,\
                   batch_size=64, drop_remainder=False )
# for i in squad_train:
#     print(i)
#     break
print("==================dataset processing complete! dataset will throw into the network!==============")

# define Models & Loss & Optimizer
char_vocab_size = len(char_vocab.vocab())
char_dim = 8
char_channel_width = 5
char_channel_size = 100
hidden_size = 100
dropout = 0.2
lr = 0.5
epoch = 6
exp_decay_rate = 0.999

encoder = Encoder(char_vocab_size, char_vocab, char_dim, char_channel_size, char_channel_width, word_vocab,
                  word_embeddings, hidden_size, dropout)                  
head = Head(hidden_size, dropout)
net = BiDAF(encoder, head)


class Loss(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, logit1, logit2, s_idx, e_idx):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logit1, s_idx) + loss_fn(logit2, e_idx)
        return loss

loss = Loss()
optimizer = nn.Adadelta(net.trainable_params(), learning_rate=lr)

trainer = Trainer(network=net, train_dataset=squad_train, loss_fn=loss, optimizer=optimizer)
trainer.run(tgt_columns=["s_idx", "e_idx"], jit=True)
  
print("Done!")




