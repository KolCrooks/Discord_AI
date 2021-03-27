import tensorflow as tf
from tensorflow import keras
import numpy as np 
from model import MessageModelStep, MessageModel
import data_manager

DATA_SET = "messages.csv"
LARGE_VOCAB_SIZE = 32
TRAIN_TEST_SPLIT = 0.75

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_m = data_manager.DataManager(file=DATA_SET, skip_first_line=False, large_vocab=LARGE_VOCAB_SIZE, train_split=TRAIN_TEST_SPLIT)

model = MessageModel(len(data_m.get_vocab()))
optimizer = keras.optimizers.Adam(learning_rate=0.0001)


ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, './logs/checkpoints', max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)

# set up the model
x_train, _ = list(data_m.sample_test(1).as_numpy_iterator())[0]
model(np.expand_dims(x_train, axis=0))

stepModel = MessageModelStep(model, data_m.decode, data_m.encode, temperature=0.01)



def generateMessage(MAX_CHARACTERS_IN_MESSAGE = 1000, starter = None, temp = 0.01):
    if starter == None:
        next_char = data_m.get_a_starter()
    else:
        next_char = starter
    result = next_char

    states = None
    for i in range(MAX_CHARACTERS_IN_MESSAGE):
        next_char, states = stepModel.generate_one_step(next_char, states=states, temp= temp)
        result += next_char
        if next_char == "\x03":
            break
    if (starter == None) and (i == MAX_CHARACTERS_IN_MESSAGE - 1):
        return generateMessage(MAX_CHARACTERS_IN_MESSAGE)
    else:
        return result