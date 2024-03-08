from functools import partial
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.legacy import SGD
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

num_samples = 1000


def kl_approx(q, p, q_tensor):
    """
    Approximates the KL divergence between two distributions.

    Args:
        q (tf.distributions.Distribution): The first distribution.
        p (tf.distributions.Distribution): The second distribution.
        q_tensor (tf.Tensor): The tensor to evaluate the log probabilities.

    Returns:
        tf.Tensor: The mean KL divergence between the distributions for the given tensor.
    """
    
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))

def divergence_fn(q, p, q_tensor, num_samples=num_samples):
    """
    Normalizes the KL divergence approximation by the number of samples.

    Args:
        q (tf.distributions.Distribution): The first distribution.
        p (tf.distributions.Distribution): The second distribution.
        q_tensor (tf.Tensor): The tensor to evaluate the log probabilities.
        num_samples (int): The number of samples for normalization.

    Returns:
        tf.Tensor: The normalized KL divergence.
    """
    return kl_approx(q, p, q_tensor) / num_samples


def prior(dtype, shape, name, trainable, add_variable_fn):
    """
    Creates an Independent multivariate normal distribution as a prior.

    Args:
        dtype: The data type of the distribution's parameters.
        shape: The shape of the distribution.
        name: The name of the distribution.
        trainable: Whether the variables are trainable.
        add_variable_fn: Function to add variables to the distribution.

    Returns:
        tfd.Independent: The Independent multivariate normal distribution.
    """
    dist = tfd.MultivariateNormalDiag(loc=1.2 * tf.ones(shape),
                                      scale_diag=3.0*tf.ones(shape))
    
    batch_ndims = tf.size(dist.batch_shape_tensor())
    
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

adjusted_divergence_fn = partial(divergence_fn, num_samples=num_samples)


def get_convolution_reparameterization(filters, kernel_size, activation, strides = 1,
                                        padding = 'SAME',
                                        prior = prior,
                                        divergence_fn = adjusted_divergence_fn,
                                        name = None) -> tfpl.Convolution2DReparameterization:
    """
    Creates a Convolution2DReparameterization layer.

    Args:
        filters (int): The number of filters.
        kernel_size (int or tuple/list of 2 ints): The kernel size.
        activation: Activation function.
        strides (int or tuple/list of 2 ints, optional): The strides of the convolution.
        padding (str, optional): The padding method.
        prior: The prior distribution function.
        divergence_fn: The divergence function for regularization.
        name (str, optional): The name of the layer.

    Returns:
        tfpl.Convolution2DReparameterization: The convolution layer with reparameterization.
    """
    return tfpl.Convolution2DReparameterization(
            filters = filters,
            kernel_size = kernel_size,
            activation = activation,
            strides = strides,
            padding = padding,
            
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_prior_fn=prior,
            kernel_divergence_fn=adjusted_divergence_fn,

            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=prior,
            bias_divergence_fn=adjusted_divergence_fn,
            name=name)


def residual_block(x, filters, kernel_size, padding, activation, pool_size, strides,  name):
    """
    Constructs a residual block with convolutional reparameterization layers.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the convolutions.
        kernel_size (int or tuple/list of 2 ints): Size of the convolution kernels.
        padding (str): Padding method.
        activation: Activation function.
        pool_size (tuple/list of 2 ints): Pool size for max pooling.
        strides (tuple/list of 2 ints): Strides for the convolution.
        name (str): Base name for the block's layers.

    Returns:
        tf.Tensor: The output tensor of the residual block.
    """
    shortcut = x

    # First convolution layer with reparameterization
    x = get_convolution_reparameterization(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=strides,
        padding=padding,
        name=name + '_conv1'
    )(x)
    x = layers.BatchNormalization(name=name + '_batchnorm1')(x)
    x = layers.ReLU()(x)

    # Second convolution layer with reparameterization
    x = get_convolution_reparameterization(
        filters=filters,
        kernel_size=kernel_size,
        activation=None,  # No activation here, it's applied after adding the shortcut
        strides=1,  # Typically, stride is set to 1 for the second conv in a residual block
        padding=padding,
        name=name + '_conv2'
    )(x)
    x = layers.BatchNormalization(name=name + '_batchnorm2')(x)

    # Adjusting the shortcut path to match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = get_convolution_reparameterization(
            filters=filters,
            kernel_size=1,  # 1x1 convolution for dimension matching
            activation=None,
            strides=1,
            padding=padding,
            name=name + '_shortcut_conv'
        )(shortcut)
        shortcut = layers.BatchNormalization(name=name + '_shortcut_batchnorm')(shortcut)

    # Merge shortcut and main path
    x = layers.Add(name=name + '_merge')([x, shortcut])
    x = layers.ReLU()(x)

    return x


def bayes_model(input_shape=(200,200,3)):
    """
    Constructs a Bayesian convolutional neural network model.

    Args:
        input_shape (tuple, optional): Shape of the input images.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    
    inputs = layers.Input(shape=input_shape, name='inputs')
    
    x = get_convolution_reparameterization(16, 3, 'swish')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    num_blocks = 3  # Increase the number of residual blocks
    filters = [32, 64, 128, 256, 512]   # Increase the number of filters in each block
    
    for i in range(num_blocks):
        x = residual_block(x, filters[i], kernel_size = 3,
                          padding = 'same', activation = tf.nn.silu,
                          pool_size = (2, 2), strides = (1, 1),
                          name = 'residual_block'+str(i))

    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    ''' 
    x = tfpl.DenseReparameterization(
        units=128,  # This matches the number of units from the Dense layer
        activation='relu',  # Activation can be directly specified here
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_divergence_fn=adjusted_divergence_fn,
        bias_divergence_fn=adjusted_divergence_fn,
        name='dense_reparam1'
        )(x)
        
    x = tfpl.DenseReparameterization(
        units=64,  # This matches the number of units from the Dense layer
        activation='relu',  # Activation can be directly specified here
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_divergence_fn=adjusted_divergence_fn,
        bias_divergence_fn=adjusted_divergence_fn,
        name='dense_reparam2'
        )(x)
    '''           
    x = tfpl.DenseReparameterization(
        units = tfpl.CategoricalMixtureOfOneHotCategorical.params_size(3, 5), activation = None,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,  
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=adjusted_divergence_fn,
        bias_divergence_fn=adjusted_divergence_fn,
        name = 'dense_reparam3')(x)

    x = tfpl.CategoricalMixtureOfOneHotCategorical(event_size = 3, num_components = 5, name = 'output')(x)   

    model = models.Model(inputs, outputs=x, name='Rice_BNN')
    
    return model
