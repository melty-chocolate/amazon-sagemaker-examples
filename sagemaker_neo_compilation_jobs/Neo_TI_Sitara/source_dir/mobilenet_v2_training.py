from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ------------------------------------------------------------ #
# Neo host methods                                             #
# ------------------------------------------------------------ #  

def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io
    import PIL.Image
    def _read_input_shape(signature):
        shape = signature[-1]['shape']
        shape[0] = 1
        return shape

    def _transform_image(image, shape_info):
        # Fetch image size
        input_shape = _read_input_shape(shape_info)

        # Perform color conversion
        if input_shape[-1] == 3:
            # training input expected is 3 channel RGB
            image = image.convert('RGB')
        elif input_shape[-1] == 1:
            # training input expected is grayscale
            image = image.convert('L')
        else:
            # shouldn't get here
            raise RuntimeError('Wrong number of channels in input shape')

        # Resize
        image = np.asarray(image.resize((input_shape[-3], input_shape[-2])))

        # Normalize
        image = image*2/255.0 - 1

        # Transpose
        if len(image.shape) == 2:  # for greyscale image
            image = np.expand_dims(image, axis=0)

        return image
 
    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'image/jpeg':
        raise RuntimeError('Content type must be image/jpeg')

    shape_info = [{"shape":[1,224,224,3], "name":"data"}]
    f = io.BytesIO(payload)
    dtest = _transform_image(PIL.Image.open(f), shape_info)
    return dtest

    
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')
    
    assert len(result) == 1
    content_type = 'application/json'
    return [json.dumps(np.squeeze(result[0]).tolist())], content_type

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #  

import os
import tensorflow as tf


INPUT_TENSOR_NAME = "input"
SIGNATURE_NAME = "serving_default"

HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 101
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50
BATCH_SIZE = 25

_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 25
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE


def model_fn(features, labels, mode, params):
    """Model function"""
    import mobilenet_v2
    
    inputs = features[INPUT_TENSOR_NAME]
    
    inputs = tf.reshape(inputs, [-1, HEIGHT, WIDTH, DEPTH])
    
    is_training = mode != tf.estimator.ModeKeys.PREDICT
    
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
        logits, endpoints = mobilenet_v2.mobilenet(inputs, NUM_CLASSES)
    
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, 
        onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(_BATCHES_PER_EPOCH * epoch) for epoch in [10, 15, 20]]
        values = [_INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=_INITIAL_LEARNING_RATE)

        #optimizer = tf.train.MomentumOptimizer(
        #    learning_rate=learning_rate,
        #    momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 224, 224, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def train_input_fn(training_dir, params):
    return _input_from_files(tf.estimator.ModeKeys.TRAIN,
                    batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, params):
    return _input_from_files(tf.estimator.ModeKeys.EVAL,
                    batch_size=BATCH_SIZE, data_dir=training_dir)


def _input_from_files(mode, batch_size, data_dir):
    """Load caltech101 deta set
  Args:
    mode: Standard names for model modes (tf.estimators.ModeKeys).
    batch_size: The number of samples per batch of input requested.
  """
    dataset = _record_dataset(_filenames(mode, data_dir))

    dataset = dataset.map(_dataset_parser)

    # For training, preprocess the image and shuffle.
    if mode == tf.estimator.ModeKeys.TRAIN:

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)
        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.
        
        # Only go through the data once.
        num_repeat = 1
        
    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    iterator = dataset.batch(batch_size).make_one_shot_iterator()
    images, labels = iterator.get_next()

    return {INPUT_TENSOR_NAME: images}, labels

def _dataset_parser(record):
    # Define the features to be parsed out of each example.
    #    You should recognize this from when we wrote the TFRecord files!
    features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    # Parse the features out of this one record we were passed
    parsed = tf.parse_single_example(record, features)
    # Format the data
    H = tf.cast(parsed['height'], tf.int32)
    W = tf.cast(parsed['width'], tf.int32)
    D = tf.cast(parsed['depth'], tf.int32)

    # Decode from the bytes that were written to the TFRecord
    image = tf.decode_raw(parsed["image"], tf.uint8)

    # Use the metadata we wrote to reshape as an image
    image = tf.reshape(image, [H, W, D])

    # Cast so we can later pass this data to convolutional layers
    image = (tf.cast(image, tf.float32) - 118) / 85 # Pre-computed mean and std

    # Crop/pad such that all images are the same size -- this will be specified in params later
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)
    
    # Tell TensorFlow what the shape is so it doesn't think this is still a dynamic variable
    image.set_shape([HEIGHT, WIDTH, DEPTH])
    
    label = tf.cast(parsed['label'], tf.int32)
    
    return image, tf.one_hot(label, NUM_CLASSES)
    
def _record_dataset(filenames):
    """Returns an input pipeline Dataset from `filenames`."""
    return tf.data.TFRecordDataset(filenames)

def _filenames(mode, data_dir):
    """Returns a list of filenames based on 'mode'."""

    if mode == tf.estimator.ModeKeys.TRAIN:
        return [os.path.join(data_dir, 'train.tfrecord')]
    elif mode == tf.estimator.ModeKeys.EVAL:
        return [os.path.join(data_dir, 'valid.tfrecord')]
    else:
        raise ValueError('Invalid mode: %s' % mode)
