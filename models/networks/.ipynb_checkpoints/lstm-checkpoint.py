import tensorflow as tf

def LSTM_3(sizes = [64, 32, 32], input_shape = (15, 15), output_shape=15):

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(sizes[0],
                                              return_sequences=True,
                                              input_shape=input_shape))
    single_step_model.add(tf.keras.layers.LSTM(sizes[1], return_sequences=True, activation='relu'))
    single_step_model.add(tf.keras.layers.LSTM(sizes[2], activation='relu'))
    single_step_model.add(tf.keras.layers.Dense(output_shape))
    
    return single_step_model