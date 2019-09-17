#! -*- coding:utf-8 -*-

import json
import numpy as np
from tqdm import tqdm
import jieba_fast as jieba
from gensim.models import Word2Vec
import re, os
import codecs
import editdistance
import warnings
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告
jieba.initialize()


mode = 0
char_size = 128
maxlen = 256
min_count = 16


word2vec = Word2Vec.load('../word2vec_baike/word2vec_baike')


id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])

for w in word2id:
    if w not in jieba.dt.FREQ:
        jieba.add_word(w)


def tokenize(s):
    return jieba.lcut(s, HMM=False)


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


webqa_data = json.load(open('../datasets/WebQA.json'))
sogou_data = json.load(open('../datasets/SogouQA.json'))


if not os.path.exists('../dgcnn_config.json'):
    chars = {}
    for D in [webqa_data, sogou_data]:
        for d in tqdm(iter(D)):
            for c in d['question']:
                chars[c] = chars.get(c, 0) + 1
            for p in d['passages']:
                for c in p['passage']:
                    chars[c] = chars.get(c, 0) + 1
    chars = {i: j for i,j in chars.items() if j >= min_count}
    id2char = {i+2: j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j: i for i,j in id2char.items()}
    json.dump([id2char, char2id], open('../dgcnn_config.json', 'w'))
else:
    id2char, char2id = json.load(open('../dgcnn_config.json'))


if not os.path.exists('../random_order.json'):
    random_order = range(len(sogou_data))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order.json'))


train_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 != mode]
dev_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 == mode]
train_data.extend(train_data)
train_data.extend(webqa_data) # 将SogouQA和WebQA按2:1的比例混合


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator(object):
    def __init__(self, data, batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def random_generate(self, s):
        l = maxlen // 2 + maxlen % 2
        if len(s) > l:
            p = np.random.random()
            if p > 0.5:
                i = np.random.randint(len(s) - l + 1)
                j = np.random.randint(l + i, min(len(s), i + maxlen) + 1)
                return s[i: j]
            else:
                return s[: maxlen]
        else:
            return s
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            Q1, Q2, P1, P2, A1, A2 = [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                # 问题
                q_text = d['question']
                q_text_words = tokenize(q_text)
                q_text = ''.join(q_text_words)
                qid = [char2id.get(c, 1) for c in q_text]
                # 篇章
                pi = np.random.choice(len(d['passages']))
                p = d['passages'][pi]
                p_text = self.random_generate(p['passage'])
                p_text_words = tokenize(p_text)
                p_text = ''.join(p_text_words)
                pid = [char2id.get(c, 1) for c in p_text]
                # 答案
                a1, a2 = np.zeros(len(p_text)), np.zeros(len(p_text))
                if p['answer']:
                    for j in re.finditer(re.escape(p['answer']), p_text):
                        a1[j.start()] = 1
                        a2[j.end() - 1] = 1
                # 组合
                Q1.append(qid)
                Q2.append(q_text_words)
                P1.append(pid)
                P2.append(p_text_words)
                A1.append(a1)
                A2.append(a2)
                if len(Q1) == self.batch_size or i == idxs[-1]:
                    Q1 = seq_padding(Q1)
                    Q2 = sent2vec(Q2)
                    P1 = seq_padding(P1)
                    P2 = sent2vec(P2)
                    A1 = seq_padding(A1)
                    A2 = seq_padding(A2)
                    yield [Q1, Q2, P1, P2, A1, A2], None
                    Q1, Q2, P1, P2, A1, A2 = [], [], [], [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from radam import RAdam


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class AttentionPooling1D(OurLayer):
    """通过加性Attention，将向量序列融合为一个定长向量
    """
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if self.h_dim is None:
            self.h_dim = input_shape[0][-1]
        self.k_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o_dense = Dense(1, use_bias=False)
    def call(self, inputs):
        xo, mask = inputs
        x = xo
        x = self.reuse(self.k_dense, x)
        x = self.reuse(self.o_dense, x)
        x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        return K.sum(x * xo, 1)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])


class DilatedGatedConv1D(OurLayer):
    """膨胀门卷积（DGCNN）
    """
    def __init__(self,
                 o_dim=None,
                 k_size=3,
                 rate=1,
                 skip_connect=True,
                 drop_gate=None,
                 **kwargs):
        super(DilatedGatedConv1D, self).__init__(**kwargs)
        self.o_dim = o_dim
        self.k_size = k_size
        self.rate = rate
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
    def build(self, input_shape):
        super(DilatedGatedConv1D, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[0][-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.rate,
            padding='same'
        )
        if self.skip_connect and self.o_dim != input_shape[0][-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)
    def call(self, inputs):
        xo, mask = inputs
        x = xo * mask
        x = self.reuse(self.conv1d, x)
        x, g = x[..., :self.o_dim], x[..., self.o_dim:]
        if self.drop_gate is not None:
            g = K.in_train_phase(K.dropout(g, self.drop_gate), g)
        g = K.sigmoid(g)
        if self.skip_connect:
            if self.o_dim != K.int_shape(xo)[-1]:
                xo = self.reuse(self.conv1d_1x1, xo)
            return (xo * (1 - g) + x * g) * mask
        else:
            return x * g * mask
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.o_dim,)


class MixEmbedding(OurLayer):
    """混合Embedding
    输入字id、词embedding，然后字id自动转字embedding，
    词embedding做一个dense，再加上字embedding，并且
    加上位置embedding。
    """
    def __init__(self, i_dim, o_dim, **kwargs):
        super(MixEmbedding, self).__init__(**kwargs)
        self.i_dim = i_dim
        self.o_dim = o_dim
    def build(self, input_shape):
        super(MixEmbedding, self).build(input_shape)
        self.char_embeddings = Embedding(self.i_dim, self.o_dim)
        self.word_dense = Dense(self.o_dim, use_bias=False)
    def call(self, inputs):
        x1, x2 = inputs
        x1 = self.reuse(self.char_embeddings, x1)
        x2 = self.reuse(self.word_dense, x2)
        return x1 + x2
    def compute_output_shape(self, input_shape):
        return input_shape[0] + (self.o_dim,)


def seq_and_vec(x):
    x, v = x
    v = K.expand_dims(v, 1)
    v = K.tile(v, [1, K.shape(x)[1], 1])
    return K.concatenate([x, v], 2)


q1_in = Input(shape=(None,)) # 问题字id输入
q2_in = Input(shape=(None, word_size)) # 问题词向量输入
p1_in = Input(shape=(None,)) # 篇章字id输入
p2_in = Input(shape=(None, word_size)) # 篇章词向量输入
a1_in = Input(shape=(None,)) # 答案左边界输入
a2_in = Input(shape=(None,)) # 答案右边界输入

q1, q2, p1, p2, a1, a2 = q1_in, q2_in, p1_in, p2_in, a1_in, a2_in
q_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(q1)
p_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(p1)

embeddings = MixEmbedding(len(char2id)+2, char_size)

q = embeddings([q1, q2])
q = Dropout(0.1)(q)
p = embeddings([p1, p2])
p = Dropout(0.1)(p)

q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, q_mask])
q = DilatedGatedConv1D(rate=2, drop_gate=0.1)([q, q_mask])
q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, q_mask])
qv = AttentionPooling1D()([q, q_mask])

p = Lambda(seq_and_vec)([p, qv])
p = Dense(char_size, use_bias=False)(p)
p = DilatedGatedConv1D(rate=1, drop_gate=0.1)([p, p_mask])
p = DilatedGatedConv1D(rate=2, drop_gate=0.1)([p, p_mask])
p = DilatedGatedConv1D(rate=4, drop_gate=0.1)([p, p_mask])
p = DilatedGatedConv1D(rate=8, drop_gate=0.1)([p, p_mask])
p = DilatedGatedConv1D(rate=16, drop_gate=0.1)([p, p_mask])
p = DilatedGatedConv1D(rate=1, drop_gate=0.1)([p, p_mask])
p =  Lambda(seq_and_vec)([p, qv])
pv = AttentionPooling1D()([p, p_mask])

pa = Dense(1, activation='sigmoid')(pv)
pa1 = Dense(1, activation='sigmoid')(p)
pa2 = Dense(1, activation='sigmoid')(p)
pa1 = Lambda(lambda x: x[0] * x[1][..., 0])([pa, pa1])
pa2 = Lambda(lambda x: x[0] * x[1][..., 0])([pa, pa2])


model = Model([q1_in, q2_in, p1_in, p2_in], [pa1, pa2])
model.summary()

train_model = Model([q1_in, q2_in, p1_in, p2_in, a1_in, a2_in], [pa1, pa2])

def focal_loss(y_true, y_pred):
    alpha, gamma = 0.25, 2
    y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
    return - alpha * y_true * K.log(y_pred) * (1 - y_pred)**gamma\
           - (1 - alpha) * (1 - y_true) * K.log(1 - y_pred) * y_pred**gamma

loss1 = focal_loss(a1_in, pa1)
loss1 = K.sum(loss1 * p_mask[..., 0]) / K.sum(p_mask)
loss2 = focal_loss(a2_in, pa2)
loss2 = K.sum(loss2 * p_mask[..., 0]) / K.sum(p_mask)
loss = (loss1 + loss2) * 100 # 放大100倍，可读性好些，不影响Adam的优化

train_model.add_loss(loss)
train_model.compile(optimizer=RAdam(1e-3))


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()


def extract_answer(q_text, p_texts, maxlen=12, threshold=0.1):
    """q_text为问题，p_texts为篇章集合（list）
    最终输出一个dict，dict的key为候选答案，而value为对应的分数。
    """
    Q1, Q2, P1, P2 = [], [], [], []
    # 问题
    q_text_words = tokenize(q_text)
    q_text = ''.join(q_text_words)
    qid = [char2id.get(c, 1) for c in q_text]
    for i, p_text in enumerate(p_texts):
        # 篇章
        p_text_words = tokenize(p_text)
        p_text = ''.join(p_text_words)
        pid = [char2id.get(c, 1) for c in p_text]
        Q1.append(qid)
        Q2.append(q_text_words)
        P1.append(pid)
        P2.append(p_text_words)
    # 给出结果序列
    Q1 = seq_padding(Q1)
    Q2 = sent2vec(Q2)
    P1 = seq_padding(P1)
    P2 = sent2vec(P2)
    A1, A2 = model.predict([Q1, Q2, P1, P2])
    # 输出每个篇章的答案
    Result = []
    for a1, a2, p in zip(A1, A2, p_texts):
        a1, a2 = a1[: len(p)], a2[: len(p)]
        l_idxs = np.where(a1 > threshold)[0]
        r_idxs = np.where(a2 > threshold)[0]
        result = {}
        for i in l_idxs:
            cond = (r_idxs >= i) & (r_idxs < i + maxlen)
            for j in r_idxs[cond]:
                k = p[i: j + 1]
                result[k] = max(result.get(k, 0), a1[i] * a2[j])
        if result:
            Result.append(result)
    # 综合所有答案
    R = {}
    for result in Result:
        for k, v in result.items():
            R[k] = R.get(k, []) + [v]
    R = {
        k: (np.array(v)**2).sum() / (sum(v) + 1)
        for k, v in R.items()
    }
    return R


def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]


def predict(data, filename, threshold=0.1):
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = extract_answer(q_text, p_texts, threshold=threshold)
            a = max_in_dict(a)
            if a:
                s = u'%s\t%s\n' % (d['id'], a)
            else:
                s = u'%s\t\n' % (d['id'])
            f.write(s)


class Evaluate(Callback):
    def __init__(self):
        self.metrics = []
        self.best = 0.
        self.stage = 0
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        acc, f1, final = self.evaluate()
        self.metrics.append((epoch, acc, f1, final))
        json.dump(self.metrics, open('train.log', 'w'), indent=4)
        if final > self.best:
            self.best = final
            train_model.save_weights('best_model.weights')
        print('learning rate: %s' % (K.eval(self.model.optimizer.lr)))
        print('acc: %.4f, f1: %.4f, final: %.4f, best final: %.4f\n' % (acc, f1, final, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 30 or (
            self.stage == 0 and epoch > 15 and
            (final < 0.5 or np.argmax(self.metrics, 0)[3] < len(self.metrics) - 5)
        ):
            """达到30个epoch，或者final开始下降到0.5以下（开始发散），
            或者连续5个epoch都没提升，就降低学习率。
            """
            self.stage = 1
            train_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
    def evaluate(self, threshold=0.1):
        predict(dev_data, 'tmp_result.txt', threshold=threshold)
        acc, f1, final = json.loads(
            os.popen(
                'python ../evaluate_tool/evaluate.py tmp_result.txt tmp_output.txt'
            ).read().strip()
        )
        return acc, f1, final


train_D = data_generator(train_data)
evaluator = Evaluate()


if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('best_model.weights')
