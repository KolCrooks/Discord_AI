import tensorflow as tf
from tensorflow import keras
import datetime
import numpy as np
import math

from tensorflow.python.training.checkpoint_management import CheckpointManager

from model import MessageModelStep, MessageModel
import data_manager

# Training Settings
EPOCHS = 300
BATCH_SIZE = 500

# Data settings
DATA_SET = "messages.csv"
SKIP_CSV_HEADERS = False
TRAIN_TEST_SPLIT = 0.75
LARGE_VOCAB_SIZE = 32
MAX_CHARACTERS_IN_MESSAGE = 1000

# Set up device
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_m = data_manager.DataManager(file=DATA_SET, skip_first_line=SKIP_CSV_HEADERS, large_vocab=LARGE_VOCAB_SIZE, train_split=TRAIN_TEST_SPLIT)
model = MessageModel(len(data_m.get_vocab()))
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)



# Logging stuff

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Checkpoints
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, './logs/checkpoints', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)


# @tf.function(experimental_relax_shapes=True)
def train_step(x_train,y_train):
    with tf.GradientTape() as tape:
        preds, state = model(x_train, training=True)
        loss_val = loss(y_train, preds)

    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss_val)
    train_accuracy(y_train, preds)

# @tf.function(experimental_relax_shapes=True)
def test_step(x_test,y_test):
    preds = model(x_test)
    loss_val = loss(y_test, preds[0])

    test_loss(loss_val)
    test_accuracy(y_test, preds[0])

for epoch in range(EPOCHS):
    for (x_train, y_train) in data_m.sample_train(n=BATCH_SIZE):
        train_step(np.expand_dims(x_train, axis=0), np.expand_dims(y_train, axis=0))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for (x_test, y_test) in data_m.sample_test():
        test_step(np.expand_dims(x_test, axis=0), np.expand_dims(y_test, axis=0))
    
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        stepModel = MessageModelStep(model, data_m.decode, data_m.encode, temperature=0.01)
        
        next_char = data_m.get_a_starter()
        result = next_char

        states = None
        for i in range(MAX_CHARACTERS_IN_MESSAGE):
            next_char, states = stepModel.generate_one_step(next_char, states=states)
            result += next_char
            if next_char == "\x03":
                break

        tf.summary.text('example', result, step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                            train_loss.result(), 
                            train_accuracy.result()*100,
                            test_loss.result(), 
                            test_accuracy.result()*100))
    if math.isnan(train_loss.result()):
        print('GRADIENT IS GONE!!!! EXITING!!!!')
        exit()
    if epoch % 10 == 0:
        manager.save()
    test_loss.reset_states()
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
