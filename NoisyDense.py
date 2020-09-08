import tensorflow as tf


class NoisyNetDense(tf.keras.layers.Layer):
    """
    A modified fully-connected layer that injects noise into the parameter distribution
    before each prediction. This randomness forces the agent to explore - at least
    until it can adjust its parameters to learn around it.
    To use: replace Dense layers (like the classifier at the end of a DQN model)
    with NoisyNetDense layers and set your policy to GreedyQ.
    See examples/noisynet_pdd_dqn_atari.py
    Reference: https://arxiv.org/abs/1706.10295
    """

    def __init__(self,
                 units,
                 activation=None,
                 **kwargs):
        super(NoisyNetDense, self).__init__(**kwargs)

        self.activation = tf.keras.activations.get(activation)
        self.units = units

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])

        # See section 3.2 of Fortunato et al.
        sqr_inputs = self.input_dim ** (1 / 2)
        self.sigma_initializer = tf.constant_initializer(value=.5 / sqr_inputs)
        self.mu_initializer = tf.random_uniform_initializer(minval=(-1 / sqr_inputs), maxval=(1 / sqr_inputs))

        self.mu_weight = self.add_weight(shape=(self.input_dim, self.units),
                                         initializer=self.mu_initializer,
                                         name='mu_weights',)

        self.sigma_weight = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=self.sigma_initializer,
                                            name='sigma_weights',)

        self.mu_bias = self.add_weight(shape=(self.units,),
                                       initializer=self.mu_initializer,
                                       name='mu_bias',
)

        self.sigma_bias = self.add_weight(shape=(self.units,),
                                          initializer=self.sigma_initializer,
                                          name='sigma_bias',)

        super(NoisyNetDense, self).build(input_shape=input_shape)

    def call(self, x, **kwargs):
        # sample from noise distribution
        e_i = tf.random.normal((self.input_dim, self.units))
        e_j = tf.random.normal((self.units,))

        # We use the factorized Gaussian noise variant from Section 3 of Fortunato et al.
        eW = tf.math.sign(e_i) * (tf.math.sqrt(tf.math.abs(e_i))) * tf.math.sign(e_j) * (tf.math.sqrt(tf.math.abs(e_j)))
        eB = tf.math.sign(e_j) * (tf.math.abs(e_j) ** (1 / 2))

        # See section 3 of Fortunato et al.
        noise_injected_weights = tf.tensordot(x, self.mu_weight + (self.sigma_weight * eW), axes=1)
        noise_injected_bias = self.mu_bias + (self.sigma_bias * eB)
        output = tf.nn.bias_add(noise_injected_weights, noise_injected_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'mu_initializer': tf.keras.initializers.serialize(self.mu_initializer),
            'sigma_initializer': tf.keras.initializers.serialize(self.sigma_initializer),
        }
        base_config = super(NoisyNetDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
