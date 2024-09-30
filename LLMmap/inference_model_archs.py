import tensorflow as tf
import numpy as np

DEFAULT_HP = {
    'num_blocks' : 3,
    'feature_size' : 384,
    'norm_layer' : tf.keras.layers.BatchNormalization,
    'num_heads' : 4,
    'activation' : 'gelu',
    'optimizer' : (tf.keras.optimizers.Adam, {'learning_rate':0.0001}),
    'with_add_dense_class' : False,
    'emb_size' : 1024,
    'num_queries' : 8,
    'num_classes' : 42,
}


class ClassTokenLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, **kwargs):
        super(ClassTokenLayer, self).__init__(**kwargs)
        self.feature_size = feature_size

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=(1, self.feature_size),
            initializer='random_normal',
            trainable=True,
            name='class_token'
        )
        super(ClassTokenLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token_tiled = tf.tile(self.class_token[tf.newaxis], (batch_size, 1, 1))
        return class_token_tiled


def transformer_block(x, hparams):
    feature_size = hparams['feature_size']
    norm = hparams['norm_layer']
    num_heads = hparams['num_heads']
    activation = hparams['activation']
    xnorm = norm()(x)
    xatt = tf.keras.layers.MultiHeadAttention(
        num_heads,
        feature_size
    )(xnorm, xnorm)
    x = tf.keras.layers.Add()([x, xatt])
    xnorm = norm()(x)
    m = tf.keras.layers.Dense(feature_size, activation=activation)(xnorm)
    x = tf.keras.layers.Add()([x, m])
    return x
    
    
def make_inference_model(hparams=DEFAULT_HP, is_for_siamese=False):
    num_classes = hparams['num_classes']
    num_queries = hparams['num_queries']
    feature_size = hparams['feature_size']
    activation = hparams['activation']
    
    traces = tf.keras.Input(shape=(num_queries, hparams['emb_size']*2), name='inputs')
    
    # special token for output
    class_token = ClassTokenLayer(feature_size, name='class_token_layer')(traces) 
        
    # emb. projection to feature_size
    traces_emb = tf.keras.layers.Dense(feature_size, activation=activation)(traces)
    
    # concat traces with class_token
    x = tf.keras.layers.Concatenate(1)([class_token, traces_emb])
    
    # transform
    for _ in range(hparams['num_blocks']):
        x = transformer_block(x, hparams)
    
    # gets output on class token
    x = x[:,0]
    
    if not is_for_siamese:
        if hparams['with_add_dense_class']:
            x = tf.keras.layers.Dense(feature_size//2, activation=activation)(x)
        output = tf.keras.layers.Dense(num_classes)(x)
    else:
        output = x

    model = keras.Model(inputs=traces, outputs=output, name='InferenceModelLLMmap')

    return model