# 模型预测脚本
import numpy as np
import pandas as pd
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import *

# BERT base
config_path = 'pre_models/bert_config.json'
checkpoint_path = 'pre_models/bert_model.ckpt'
dict_path = 'pre_models/vocab.txt'

n = 5   # Cross-validation
SEED = 2020
num_classes = 14

maxlen = 512
max_segment = 2
batch_size = 4
grad_accum_steps = 64
drop = 0.2
lr = 2e-5
epochs = 100


def load_data(df):
    """加载数据。"""
    D = list()
    for _, row in df.iterrows():
        text = row['text']
        label = row['label']
        D.append((text, int(label)))
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def sentence_split(words):
    """句子截断。"""
    document_len = len(words)

    index = list(range(0, document_len, maxlen-2))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = tokenizer.tokens_to_ids(['[CLS]'] + segment + ['[SEP]'])
        segments.append(segment)

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


class data_generator(DataGenerator):
    """数据生成器。"""

    def __init__(self, data, batch_size=32, buffer_size=None, random=False):
        super().__init__(data, batch_size, buffer_size)
        self.random = random

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids = sentence_split(text)
            token_ids = sequence_padding(token_ids, length=maxlen)
            segment_ids = np.zeros_like(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(
                    batch_token_ids, length=max_segment)
                batch_segment_ids = sequence_padding(
                    batch_segment_ids, length=max_segment)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


class Attention(Layer):
    """注意力层。"""
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.truncated_normal(mean=0.0, stddev=0.05)
        # 为该层创建一个可训练的权重
        self.weight = self.add_weight(
            name='weight',
            shape=(self.hidden_size, self.hidden_size),
            initializer=initializer,
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.hidden_size,),
            initializer='zero',
            trainable=True)
        self.query = self.add_weight(
            name='query',
            shape=(self.hidden_size, 1),
            initializer=initializer,
            trainable=True)

        super().build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        x, mask = x
        mask = K.squeeze(mask, axis=2)
        # linear
        key = K.bias_add(K.dot(x, self.weight), self.bias)

        # compute attention
        outputs = K.squeeze(K.dot(key, self.query), axis=2)
        outputs -= 1e32 * (1 - mask)

        attn_scores = K.softmax(outputs)
        attn_scores *= mask
        attn_scores = K.reshape(
            attn_scores, shape=(-1, 1, attn_scores.shape[-1]))

        outputs = K.squeeze(K.batch_dot(attn_scores, key), axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.hidden_size


def build_model():
    """构建模型。"""
    token_ids = Input(shape=(max_segment, maxlen), dtype='int32')
    segment_ids = Input(shape=(max_segment, maxlen), dtype='int32')

    input_mask = Masking(mask_value=0)(token_ids)
    input_mask = Lambda(
        lambda x: K.cast(K.any(x, axis=2, keepdims=True), 'float32')
    )(input_mask)

    token_ids1 = Lambda(
        lambda x: K.reshape(x, shape=(-1, maxlen))
    )(token_ids)
    segment_ids1 = Lambda(
        lambda x: K.reshape(x, shape=(-1, maxlen))
    )(segment_ids)

    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    output = bert.model([token_ids1, segment_ids1])
    output = Lambda(lambda x: x[:, 0])(output)
    output = Lambda(
        lambda x: K.reshape(x, shape=(-1, max_segment, output.shape[-1]))
    )(output)
    output = Multiply()([output, input_mask])
    output = Dropout(drop)(output)

    output = Attention(output.shape[-1].value)([output, input_mask])
    output = Dropout(drop)(output)

    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model([token_ids, segment_ids], output)

    return model


def do_predict(df_test):
    test_data = load_data(df_test)
    test_generator = data_generator(test_data, batch_size)

    model = build_model()

    res = np.zeros((len(test_data), num_classes))
    for i in range(1, n+1):
        model.load_weights(f'weights-{i}.h5')
        pred = model.predict_generator(
            test_generator.forfit(), steps=len(test_generator))
        res += pred / n

    return res


if __name__ == '__main__':
    df_test = pd.read_csv('data/test_a.csv', sep='\t')
    df_test['label'] = 0
    df_test['text'] = df_test['text'].apply(lambda x: x.strip().split())

    res = do_predict(df_test)

    df_test['label'] = res.argmax(axis=1)
    df_test.to_csv('submission.csv', index=False, columns=['label'])
