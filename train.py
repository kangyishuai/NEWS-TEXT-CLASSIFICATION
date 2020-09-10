# 模型训练脚本

import numpy as np
import pandas as pd
from bert4keras.backend import keras, search_layer, K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
            attn_scores, shape=(-1, 1, attn_scores.shape[-1])
        )

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

    optimizer_params = {
        'learning_rate': lr,
        'grad_accum_steps': grad_accum_steps
    }
    optimizer = extend_with_gradient_accumulation(Adam)
    optimizer = optimizer(**optimizer_params)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def adversarial_training(model, embedding_name, epsilon=1.):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


class Evaluator(Callback):
    def __init__(self, valid_generator):
        super().__init__()
        self.valid_generator = valid_generator
        self.best_val_f1 = 0.

    def evaluate(self):
        y_true, y_pred = list(), list()
        for x, y in self.valid_generator:
            y_true.append(y)
            y_pred.append(self.model.predict(x).argmax(axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.best_val_f1:.5f}')


def do_train(df_train):
    skf = StratifiedKFold(n_splits=n, random_state=SEED, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text'], df_train['label']), 1):
        print(f'Fold {fold}')

        train_data = load_data(df_train.iloc[trn_idx])
        valid_data = load_data(df_train.iloc[val_idx])

        train_generator = data_generator(train_data, batch_size, random=True)
        valid_generator = data_generator(valid_data, batch_size)

        model = build_model()

        adversarial_training(model, 'Embedding-Token', 0.5)

        callbacks = [
            Evaluator(valid_generator),
            EarlyStopping(
                monitor='val_f1',
                patience=5,
                verbose=1,
                mode='max'),
            ReduceLROnPlateau(
                monitor='val_f1',
                factor=0.5,
                patience=2,
                verbose=1,
                mode='max'),
            ModelCheckpoint(
                f'weights-{fold}.h5',
                monitor='val_f1',
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
                mode='max'),
        ]

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=valid_generator.forfit(),
            validation_steps=len(valid_generator)
        )

        del model
        K.clear_session()


if __name__ == '__main__':
    df_train = pd.read_csv('data/train_set.csv', sep='\t')
    df_train['text'] = df_train['text'].apply(lambda x: x.strip().split())

    do_train(df_train)
