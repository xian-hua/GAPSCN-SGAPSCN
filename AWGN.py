import tensorflow as tf
@tf.autograph.experimental.do_not_convert
def awgn(input_signal, snr_dB, rate=1.0):
    output = tf.zeros([input_signal.shape[1], input_signal.shape[2]], tf.float32)
    snr_linear = 10 ** (snr_dB / 10.0)
    for i in range(input_signal.shape[0]):
        input = input_signal[i]
        shape = tf.dtypes.cast(tf.size(input), tf.float32)
        avg_energy = tf.reduce_mean(tf.abs(input) * tf.abs(input))
        noise_variance = avg_energy / snr_linear
        noisenormal = tf.random.normal([tf.dtypes.cast(shape, tf.int32)])
        noise = tf.sqrt(noise_variance) * noisenormal
        output_signal = input + noise
        output = tf.concat([output, output_signal], 0)
    output = output[1:]
    output = tf.reshape(output, [output.shape[0], 1, output.shape[1]])
    return output

@tf.autograph.experimental.do_not_convert
def awgn_ds(input_signal, snr_dB, rate=1.0):
    output = tf.zeros([1,input_signal.shape[1], input_signal.shape[2]], tf.float32)
    snr_linear = 10 ** (snr_dB / 10.0)
    for i in range(input_signal.shape[0]):
        input = input_signal[i]
        shape = tf.dtypes.cast(tf.size(input), tf.float32)
        avg_energy = tf.reduce_mean(tf.abs(input) * tf.abs(input))
        noise_variance = avg_energy / snr_linear
        noisenormal = tf.random.normal([tf.dtypes.cast(shape, tf.int32)])
        noisenormal=tf.expand_dims(noisenormal, axis=-1)
        noise = tf.sqrt(noise_variance) * noisenormal
        output_signal = input + noise
        output_signal=tf.expand_dims(output_signal, axis=0)
        output = tf.concat([output, output_signal], 0)
    output = output[1:]
    return output
