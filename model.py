import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class MessageModel(keras.Model):
    def __init__(self, vocab_size):
        super(MessageModel, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, 256)
        self.gru = keras.layers.GRU(1024, return_sequences=True, return_state=True)
        self.dropout = keras.layers.Dropout(0.2)
        self.dense = keras.layers.Dense(vocab_size, activation="softmax")
    
    def call(self, inputs, states = None, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)

        if training:
            x = self.dropout(x)

        x = self.dense(x,training=training)
        return x, states

class MessageModelStep(tf.keras.Model):
    def __init__(self, model, decode, encode, temperature=1.0):
        super().__init__()
        self.temperature=temperature
        self.model = model
        self.decode = decode
        self.encode = encode

    def generate_one_step(self, inputs, states=None, temp = 1.0):
        # Convert strings to token IDs.
        input_ids = np.expand_dims(self.encode(inputs), axis=0)

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits] 
        predicted_logits, states =  self.model(inputs=input_ids, states=states)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/temp

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        
        # Convert from token ids to characters
        predicted_chars = self.decode(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states