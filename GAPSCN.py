# +
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense,ReLU, LeakyReLU, Layer, Concatenate, MaxPool1D, UpSampling1D, ZeroPadding1D, Dense, GlobalAveragePooling1D,GlobalMaxPool1D,Concatenate,Permute,Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as MSE
import scipy.io
import matplotlib.pyplot as plt
from tensorflow_compression.python.layers.gdn import GDN
from LearningRateSchedule import LinearWarmup
from AWGN import awgn_ds

class RB(Layer):
    def __init__(self, neurons):
        super(RB, self).__init__()
        self.conv1 = Conv1D(neurons, 1, padding='same', activation='relu')
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv3 = Conv1D(neurons, 1, padding='same')
        self.relu = ReLU()
    
    def call(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = inputs + output
        output = self.relu(output)
        return output

class attentionE(Layer):
    def __init__(self, neuron,cneuron):
        super(attentionE, self).__init__()
        self.conv1 = Conv1D(neuron, 1, padding='same', activation='sigmoid')
        self.conv2 = Conv1D(cneuron, 1, padding='same', activation='sigmoid')
        self.conv2d = Conv2D(1, 1, padding='same', activation='sigmoid')
        self.RB1 = RB(neuron)
        self.RB2 = RB(neuron)
        self.RB3 = RB(neuron)
        self.RB4 = RB(neuron)
        self.RB5 = RB(neuron)
        self.RB6 = RB(neuron)
        self.GDN = GDN()
        self.permute = Permute((2,1))
        self.permute1 = Permute((2,1))
    
    def call(self, inputs):
        output1 = self.RB1(inputs)
        output1 = self.RB2(output1)       
        output1 = self.RB3(output1)

        output2 = self.conv1(output1)        
        output2 = tf.math.multiply(output1, output2)

        output3p = self.permute(output1)
        output3 = self.conv2(output3p)        
        output3 = tf.math.multiply(output3, output3p)
        output3 = self.permute1(output3)
        
        output4 = tf.expand_dims(output1,axis=-1)
        output4 = self.conv2d(output4)
        output4 = tf.squeeze(output4,[-1])
        output4 = tf.math.multiply(output1, output4)
        
        output = (output2+output3+output4)
        output = self.GDN(output)
        
        output = inputs + output
        return output

class attentionD(Layer):
    def __init__(self, neuron, cneuron):
        super(attentionD, self).__init__()
        self.conv1 = Conv1D(neuron, 1, padding='same', activation='sigmoid')
        self.conv2 = Conv1D(cneuron, 1, padding='same', activation='sigmoid')
        self.conv2d = Conv2D(1, 1, padding='same', activation='sigmoid')
        self.RB1 = RB(neuron)
        self.RB2 = RB(neuron)
        self.RB3 = RB(neuron)
        self.RB4 = RB(neuron)
        self.RB5 = RB(neuron)
        self.RB6 = RB(neuron)
        self.relu = ReLU()
        self.IGDN = GDN(inverse=True)
        self.permute = Permute((2,1))
        self.permute1 = Permute((2,1))
    
    def call(self, inputs):
        output1 = self.RB1(inputs)
        output1 = self.RB2(output1)
        output1 = self.RB3(output1)

        output2 = self.conv1(output1)        
        output2 = tf.math.multiply(output1, output2)
               
        
        output3p = self.permute(output1)
        output3 = self.conv2(output3p)        
        output3 = tf.math.multiply(output3, output3p)
        output3 = self.permute1(output3)
    
        output4 = tf.expand_dims(output1,axis=-1)
        output4 = self.conv2d(output4)
        output4 = tf.squeeze(output4,[-1])
        output4 = tf.math.multiply(output1, output4)
        
        output = (output2+output3+output4)
        output = self.IGDN(output)
        
        output = inputs + output
        return output

class resblockE(Layer):
    def __init__(self,neurons):
        super(resblockE, self).__init__()
        self.conv1 = Conv1D(neurons, 1, padding='same')#, activation='relu'
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv3 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.ds1 = MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.ds2 = MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.relu = ReLU()
        self.GDN = GDN()

    
    def call(self, inputs, **kwargs):
        resoutput = self.conv1(inputs)        
        resoutput = self.ds1(resoutput)
        
        output = self.conv2(inputs)
        output = self.ds2(output)
        output = self.conv3(output)               
        
        
        output = resoutput + output
        output = self.GDN(output)
        return output

class resblockD(Layer):
    def __init__(self,neurons):
        super(resblockD, self).__init__()
        self.conv1 = Conv1D(neurons, 1, padding='same')#add relu, activation='relu'
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv3 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.us1 = UpSampling1D(size=2)
        self.us2 = UpSampling1D(size=2)
        self.relu = ReLU()
        self.IGDN = GDN(inverse=True)

    
    def call(self, inputs, **kwargs):
        resoutput = self.conv1(inputs)  
        resoutput = self.us1(resoutput)        
        
        output = self.conv2(inputs)
        output = self.us2(output)
        output = self.conv3(output) 
                
        output = resoutput + output   
        output = self.IGDN(output)
        return output

class Encoder(Model):
    def __init__(self, neuron):
        super(Encoder, self).__init__()
        self.attention1 = attentionE(neuron,64)
        self.attention2 = attentionE(neuron,32)
        self.res1 = resblockE(neuron)
        self.res2 = resblockE(neuron)
        self.res3 = resblockE(neuron)
        self.res4 = resblockE(neuron)
        self.GDN1 = GDN()
        self.conv = Conv1D(1, 1)
        self.dense = Dense(1)


    def call(self, inputs):
        
        output = self.res1(inputs)
        output = self.attention1(output)
        output = self.res2(output)
        output = self.attention2(output)       
        output = self.dense(output)
        output = self.GDN1(output)
        return output

class Decoder(Model):
    def __init__(self, neuron):
        super(Decoder, self).__init__()
        self.attention1 = attentionD(neuron,128)
        self.attention2 = attentionD(neuron,128)
        self.res1 = resblockD(neuron)
        self.res2 = resblockD(neuron)
        self.res3 = resblockD(neuron)
        self.res4 = resblockD(neuron)
        self.conv = Conv1D(1, 1, activation='sigmoid')
        self.dense = Dense(1,activation='sigmoid')

    def call(self, inputs): 
        
        output = self.res1(inputs)  
        output = self.attention1(output)
        output = self.res2(output)   
        output = self.attention2(output)
        output = self.dense(output)
        return output

class AE(Model):
    def __init__(self, filter):
        super(AE, self).__init__()
        self.enc = Encoder(filter)
        self.dec = Decoder(filter)

    def call(self, inputs):
        output = self.enc(inputs)
        output = awgn_ds(output,20)
        output = self.dec(output)
        return output

model = AE(64)
# model.build(input_shape = (10, 128, 1))
# model.call(tf.keras.layers.Input(shape = (128, 1)))
# model.summary()
