from keras.utils import plot_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, maximum,concatenate, Reshape, add, Dropout, Activation
from keras.models import Model
from keras import backend as K
from keras import regularizers
from params import *





###
# model used in the paper
###
def model_mixedInput_4Layers_64(outputLayerOfPartialNet = 4):
    layerDim = 64;

    inputs_V = Input(shape=(dim,dim,))
    inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
    inputs_A = Input(shape=(int(dim/2),dim*2,))
    inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
    
    
    x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
    l1 = Dense(layerDim, activation='sigmoid')(x1);
    l2 = Dense(layerDim, activation='sigmoid')(l1)
    l3 = Dense(layerDim, activation='sigmoid')(l2)
    encoded = Dense(layerDim, activation='sigmoid')(l3)
    
    
    x1 = Dense(layerDim, activation='relu')(encoded)
    x1 = Dense(layerDim, activation='relu')(x1)
    x1 = Dense(dim*dim, activation='sigmoid')(x1)
    decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
    
    x2 = Dense(layerDim, activation='relu')(encoded)
    x2 = Dense(layerDim, activation='relu')(x2)
    x2 = Dense(dim*dim, activation='sigmoid')(x2)
    decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
    
    
    model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
    if (outputLayerOfPartialNet==1):
        model_partial = Model([inputs_V,inputs_A], l1)
    elif (outputLayerOfPartialNet==2):
        model_partial = Model([inputs_V,inputs_A], l2)
    elif (outputLayerOfPartialNet==3):
        model_partial = Model([inputs_V,inputs_A], l3)
    elif (outputLayerOfPartialNet==4):
        model_partial = Model([inputs_V,inputs_A], encoded)
    
    return (model_full, model_partial);

def model_mixedInput_1Layer_64():
    layerDim = 64;
 
    inputs_V = Input(shape=(dim,dim,))
    inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
    inputs_A = Input(shape=(int(dim/2),dim*2,))
    inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
     
     
    x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
    encoded = Dense(layerDim, activation='sigmoid')(x1)
     
     
    x1 = Dense(dim*dim, activation='sigmoid')(encoded)
    decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
     
    x2 = Dense(dim*dim, activation='sigmoid')(encoded)
    decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
     
     
    model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
    model_partial = Model([inputs_V,inputs_A], encoded)
     
    return (model_full, model_partial);


def model_mixedInput_2Layers_64():
    layerDim = 64;
 
    inputs_V = Input(shape=(dim,dim,))
    inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
    inputs_A = Input(shape=(int(dim/2),dim*2,))
    inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
     
     
    x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
    l1 = Dense(layerDim, activation='sigmoid')(x1)
    encoded = Dense(layerDim, activation='sigmoid')(l1)
     
     
    x1 = Dense(layerDim, activation='relu')(encoded)
    x1 = Dense(dim*dim, activation='sigmoid')(x1)
    decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
     
    x2 = Dense(layerDim, activation='relu')(encoded)
    x2 = Dense(dim*dim, activation='sigmoid')(x2)
    decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
     
     
    model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
    model_partial = Model([inputs_V,inputs_A], encoded)
     
    return (model_full, model_partial);

def model_mixedInput_3Layers_64():
    layerDim = 64;
 
    inputs_V = Input(shape=(dim,dim,))
    inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
    inputs_A = Input(shape=(int(dim/2),dim*2,))
    inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
     
     
    x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
    l1 = Dense(layerDim, activation='sigmoid')(x1)
    l2 = Dense(layerDim, activation='sigmoid')(l1)
    encoded = Dense(layerDim, activation='sigmoid')(l2)
     
     
    x1 = Dense(layerDim, activation='relu')(encoded)
    x1 = Dense(dim*dim, activation='relu')(x1)
    x1 = Dense(dim*dim, activation='sigmoid')(x1)
    decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
     
    x2 = Dense(layerDim, activation='relu')(encoded)
    x2 = Dense(dim*dim, activation='relu')(x2)
    x2 = Dense(dim*dim, activation='sigmoid')(x2)
    decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
     
     
    model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
    model_partial = Model([inputs_V,inputs_A], encoded)
     
    return (model_full, model_partial);




def model_twoStages_4Layers_64():
    layerDim = 64;
 
    inputs_V = Input(shape=(dim,dim,))
    inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
    vl1 = Dense(layerDim, activation='sigmoid')(inputs_V_reshaped);
    vl2 = Dense(layerDim, activation='sigmoid')(vl1)
    vl3 = Dense(layerDim, activation='sigmoid')(vl2)
    
    inputs_A = Input(shape=(int(dim/2),dim*2,))
    inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
    al1 = Dense(layerDim, activation='sigmoid')(inputs_A_reshaped);
    al2 = Dense(layerDim, activation='sigmoid')(al1)
    al3 = Dense(layerDim, activation='sigmoid')(al2)
    
    x1 = concatenate([vl3, al3]);
    encoded = Dense(layerDim, activation='sigmoid')(x1)
    
    x1 = Dense(layerDim, activation='relu')(encoded)
    x1 = Dense(layerDim, activation='relu')(x1)
    x1 = Dense(dim*dim, activation='sigmoid')(x1)
    decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
    
    x2 = Dense(layerDim, activation='relu')(encoded)
    x2 = Dense(layerDim, activation='relu')(x2)
    x2 = Dense(dim*dim, activation='sigmoid')(x2)
    decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);

    model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
    model_partial = Model([inputs_V,inputs_A], encoded)
     
    return (model_full, model_partial);


###
# Model to conduct the shared representation learning from the encoded representations at the middle of the autoencoder
### 
def sharedRepLearning_encodedInput():
    layerDim = 64;

    inputs = Input(shape=(layerDim,))
#     l1 = Dense(layerDim, activation='relu')(inputs)
#     l2 = Dense(layerDim, activation='relu')(l1)
#     dropout = Dropout(0.2)(inputs)
    l1 = Dense(64, activation='relu')(inputs)
    l1 = Dropout(0.2)(l1)
    l2 = Dense(64, activation='relu')(l1)
    l2 = Dropout(0.2)(l2)
    l3 = Dense(10, activation='softmax')(l2)
    
    model_supervised = Model(inputs, l3)
    
    return model_supervised;






# def model1():
#     ## multimodal deep encoder. 
#     inputs_1 = Input(shape=(dim,dim))
#     reshaped_1 = Reshape((dim,dim,1), input_shape=(dim, dim))(inputs_1);
#     x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped_1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Activation('softmax')(x1);
#     # x1 = Dropout(0.4)(x1)
#     
#     inputs_2 = Input(shape=(dim,dim))
#     reshaped_2 = Reshape((dim,dim,1), input_shape=(dim, dim))(inputs_2);
#     x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped_2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Activation('softmax')(x2);
#     # x2 = Dropout(0.4)(x2)
#     
#     # encoded = add([x1, x2])
#     # encoded = maximum([x1, x2])
#     encoded = concatenate([x1, x2]);
#     encoded = Dense(8,input_shape=(4,4,8*2))(encoded);
#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#     
#     # x1 = Dropout(0.4)(encoded)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(16, (3, 3), activation='relu')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x1)
#     decoded_1 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x1);
#     
#     # x2 = Dropout(0.4)(encoded)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(16, (3, 3), activation='relu')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x2);
#     
#     autoencoder_comb = Model([inputs_1,inputs_2], [decoded_1,decoded_2])
#     plot_model(autoencoder_comb, show_shapes=True, to_file='data/model_auto_comb.png')
#     
#     partialNetwork = Model([inputs_1,inputs_2], encoded)
#     
#     return (autoencoder_comb, partialNetwork);
#     # autoencoder_mnist = Model(inputs_1, decoded_1)
    
    
# def model2():
#     ### model with signoid activation function implemented at the middle (for the sake of visibility at the analysis)
#     ## IMAGE
#     
#     inputs_1 = Input(shape=(dim,dim))
#     reshaped_1 = Reshape((dim,dim,1), input_shape=(dim, dim))(inputs_1);
#     x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped_1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Activation('softmax')(x1);
#     # x1 = Dropout(0.4)(x1)
#     
#     inputs_2 = Input(shape=(dim,dim))
#     reshaped_2 = Reshape((dim,dim,1), input_shape=(dim, dim))(inputs_2);
#     x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped_2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = MaxPooling2D((2, 2), padding='same')(x2)
#     x2 = Activation('softmax')(x2);
#     # x2 = Dropout(0.4)(x2)
#     
#     # encoded = add([x1, x2])
#     # encoded = maximum([x1, x2])
#     encoded = concatenate([x1, x2]);
#     encoded = Dense(8,activation='sigmoid')(encoded);
#     
#     # print("encoded"+str(encoded.get_shape))
#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#     
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(16, (3, 3), activation='relu')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x1)
#     decoded_1 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x1);
#     
#     
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(16, (3, 3), activation='relu')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x2);
#        
#     autoencoder_comb = Model([inputs_1,inputs_2], [decoded_1,decoded_2])
#     plot_model(autoencoder_comb, show_shapes=True, to_file='data/model_auto_comb_sigmoid.png')
#     
#     partialNetwork = Model([inputs_1,inputs_2], encoded)
#     
#     return (autoencoder_comb, partialNetwork);
   
   
# def model3_oneInput():
#     
#     inputs_V = Input(shape=(dim,dim))
#     inputs_A = Input(shape=(dim,dim))
#     
#     x1 = concatenate([inputs_V, inputs_A]);
#     x1 = Dense(28)(x1);
#     x1 = Reshape((dim,dim,1), input_shape=(dim, dim))(x1);
#     pool1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x1)
#     x1 = MaxPooling2D((2, 2), padding='same')(pool1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     pool2 = MaxPooling2D((2, 2), padding='same')(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
#     pool3 = MaxPooling2D((2, 2), padding='same')(x1)
#     encoded = Activation('sigmoid')(pool3);
#     # encoded = Dropout(0.4)(x1)
#     
#     # encoded = Dense(8,activation='sigmoid')(x1);
#     
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(16, (3, 3), activation='relu')(x1)
#     x1 = UpSampling2D((2, 2))(x1)
#     x1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x1)
#     decoded_1 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x1);
#     
#     
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(16, (3, 3), activation='relu')(x2)
#     x2 = UpSampling2D((2, 2))(x2)
#     x2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim, dim, 1))(x2);
#     
#     
#     
#     autoencoder_comb = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     plot_model(autoencoder_comb, show_shapes=True, to_file='data/171124_model_oneInput.png')
# 
# #     partialNetwork = Model([inputs_V,inputs_A], encoded)
#     partialNetwork = Model([inputs_V,inputs_A],encoded);
#     print("** model constructed **")
#     
#     return (autoencoder_comb, partialNetwork);
    
   
   
# def model4_dense():
#     ## IMAGE
# 
#     layerDim = 124;
#     
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_A = Input(shape=(dim,dim,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     l1 = Dense(layerDim, activation='sigmoid')(x1);
#     l2 = Dense(layerDim, activation='sigmoid')(l1)
#     encoded = Dense(layerDim, activation='sigmoid')(l2)
#     
#     
#     x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim*dim,1))(x2);
#     
#     
#     autoencoder_comb = Model([inputs_V,inputs_A], [decoded_1,decoded_2])    
#     plot_model(autoencoder_comb, show_shapes=True, to_file='171128_model_dense.png')
#     
#     
# #     partialNetwork = Model([inputs_V,inputs_A],l1);
# #     partialNetwork = Model([inputs_V,inputs_A],l2);
#     partialNetwork = Model([inputs_V,inputs_A],encoded);
#     
#     print("** model constructed **")
#     
#     return (autoencoder_comb, partialNetwork);
   
# def model5_dense_sparse():
#     ## IMAGE
# 
#     layerDim = 124;
#     
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_V_reg = Dense(dim*dim,activity_regularizer=regularizers.l1(10e-5))(inputs_V_reshaped);
#     
#     inputs_A = Input(shape=(dim,dim,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_A);
#     inputs_A_reg = Dense(dim*dim,activity_regularizer=regularizers.l1(10e-5))(inputs_A_reshaped);
#     
#     
#     x1 = concatenate([inputs_V_reg, inputs_A_reg]);
#     l1 = Dense(layerDim, activation='sigmoid',activity_regularizer=regularizers.l1(10e-5))(x1);
#     l2 = Dense(layerDim, activation='sigmoid',activity_regularizer=regularizers.l1(10e-5))(l1)
#     encoded = Dense(layerDim, activation='sigmoid',activity_regularizer=regularizers.l1(10e-5))(l2)
#     
#     
#     x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim*dim,1))(x2);
#     
#     
#     autoencoder_comb = Model([inputs_V,inputs_A], [decoded_1,decoded_2])    
#     plot_model(autoencoder_comb, show_shapes=True, to_file='171128_model_dense_sparse.png')
#     
#     
# #     partialNetwork = Model([inputs_V,inputs_A],l1);
# #     partialNetwork = Model([inputs_V,inputs_A],l2);
#     partialNetwork = Model([inputs_V,inputs_A],encoded);
#     
#     print("** model constructed **")
#     
#     return (autoencoder_comb, partialNetwork);


# def model6_dense_4layers():
#     ## IMAGE
# 
#     layerDim = 124;
#     
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     
#     inputs_A = Input(shape=(dim,dim,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     l1 = Dense(layerDim, activation='sigmoid')(x1);
#     l2 = Dense(layerDim, activation='sigmoid')(l1)
#     l3 = Dense(layerDim, activation='sigmoid')(l2)
#     encoded = Dense(layerDim, activation='sigmoid')(l3)
#     
#     
#     x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim*dim,1))(x2);
#     
#     
#     autoencoder_comb = Model([inputs_V,inputs_A], [decoded_1,decoded_2])    
#     plot_model(autoencoder_comb, show_shapes=True, to_file='171128_model_dense_sparse.png')
#     
#     
# #     partialNetwork = Model([inputs_V,inputs_A],l1);
# #     partialNetwork = Model([inputs_V,inputs_A],l2);
# #     partialNetwork = Model([inputs_V,inputs_A],l3);
#     partialNetwork = Model([inputs_V,inputs_A],encoded);
#     
#     print("** model constructed **")
#     
#     return (autoencoder_comb, partialNetwork);
   
# def model7_dense_1layer():
#     ## IMAGE
# 
#     layerDim = 124;
#     
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     
#     inputs_A = Input(shape=(dim,dim,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     encoded = Dense(layerDim, activation='sigmoid')(x1);
#     
#     
#     x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((dim,dim), input_shape=(dim*dim,1))(x2);
#     
#     
#     autoencoder_comb = Model([inputs_V,inputs_A], [decoded_1,decoded_2])    
#     plot_model(autoencoder_comb, show_shapes=True, to_file='171204_model_dense_1layer.png')
#     
#     
# #     partialNetwork = Model([inputs_V,inputs_A],l1);
# #     partialNetwork = Model([inputs_V,inputs_A],l2);
# #     partialNetwork = Model([inputs_V,inputs_A],l3);
#     partialNetwork = Model([inputs_V,inputs_A],encoded);
#     
#     print("** model constructed **")
#     
#     return (autoencoder_comb, partialNetwork);

# def model8_revisedInput_1layer():
#     layerDim = 124;
# 
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_A = Input(shape=(int(dim/2),dim*2,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     encoded = Dense(layerDim, activation='sigmoid')(x1);
#     # l2 = Dense(layerDim, activation='sigmoid')(l1)
#     # encoded = Dense(layerDim, activation='sigmoid')(l2)
#     
#     
#     # x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     # x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(encoded)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     # x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     # x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(encoded)
#     decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
#     
#     
#     model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     model_partial = Model([inputs_V,inputs_A], encoded)
#     
#     return (model_full, model_partial);
    
    
# def model8_revisedInput_3layer():
#     layerDim = 124;
# 
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_A = Input(shape=(int(dim/2),dim*2,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     l1 = Dense(layerDim, activation='sigmoid')(x1);
#     l2 = Dense(layerDim, activation='sigmoid')(l1)
#     encoded = Dense(layerDim, activation='sigmoid')(l2)
#     
#     
#     x1 = Dense(layerDim, activation='sigmoid')(encoded)
#     x1 = Dense(layerDim, activation='sigmoid')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='sigmoid')(encoded)
#     x2 = Dense(layerDim, activation='sigmoid')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
#     
#     
#     model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     model_partial = Model([inputs_V,inputs_A], encoded)
#     
#     return (model_full, model_partial);
   

# def model8_revisedInput_3layer_reluDecoder():
#     layerDim = 124;
# 
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_A = Input(shape=(int(dim/2),dim*2,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     l1 = Dense(layerDim, activation='sigmoid')(x1);
#     l2 = Dense(layerDim, activation='sigmoid')(l1)
#     encoded = Dense(layerDim, activation='sigmoid')(l2)
#     
#     
#     x1 = Dense(layerDim, activation='relu')(encoded)
#     x1 = Dense(layerDim, activation='relu')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='relu')(encoded)
#     x2 = Dense(layerDim, activation='relu')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
#     
#     
#     model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     model_partial = Model([inputs_V,inputs_A], encoded)
#     
#     return (model_full, model_partial);

# def model8_revisedInput_4layer_reluDecoder():
#     layerDim = 124;
# 
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     inputs_A = Input(shape=(int(dim/2),dim*2,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
#     
#     
#     x1 = concatenate([inputs_V_reshaped, inputs_A_reshaped]);
#     l1 = Dense(layerDim, activation='sigmoid')(x1);
#     l2 = Dense(layerDim, activation='sigmoid')(l1)
#     l3 = Dense(layerDim, activation='sigmoid')(l2)
#     encoded = Dense(layerDim, activation='sigmoid')(l3)
#     
#     
#     x1 = Dense(layerDim, activation='relu')(encoded)
#     x1 = Dense(layerDim, activation='relu')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='relu')(encoded)
#     x2 = Dense(layerDim, activation='relu')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
#     
#     
#     model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     model_partial = Model([inputs_V,inputs_A], encoded)
#     
#     return (model_full, model_partial);
# 
# 
# 
# 
# def model9_separateInputs_4layer_reluDecoder_64():
#     layerDim = 64;
#  
#     inputs_V = Input(shape=(dim,dim,))
#     inputs_V_reshaped = Reshape((dim*dim,), input_shape=(dim, dim))(inputs_V);
#     vl1 = Dense(layerDim, activation='sigmoid')(inputs_V_reshaped);
#     vl2 = Dense(layerDim, activation='sigmoid')(vl1)
#     vl3 = Dense(layerDim, activation='sigmoid')(vl2)
#     
#     inputs_A = Input(shape=(int(dim/2),dim*2,))
#     inputs_A_reshaped = Reshape((dim*dim,), input_shape=(int(dim/2), dim*2))(inputs_A);
#     al1 = Dense(layerDim, activation='sigmoid')(inputs_A_reshaped);
#     al2 = Dense(layerDim, activation='sigmoid')(al1)
#     al3 = Dense(layerDim, activation='sigmoid')(al2)
#     
#     x1 = concatenate([vl3, al3]);
#     encoded = Dense(layerDim, activation='sigmoid')(x1)
#     
#     x1 = Dense(layerDim, activation='relu')(encoded)
#     x1 = Dense(layerDim, activation='relu')(x1)
#     x1 = Dense(dim*dim, activation='sigmoid')(x1)
#     decoded_1 =  Reshape((dim,dim), input_shape=(dim*dim,1))(x1);
#     
#     x2 = Dense(layerDim, activation='relu')(encoded)
#     x2 = Dense(layerDim, activation='relu')(x2)
#     x2 = Dense(dim*dim, activation='sigmoid')(x2)
#     decoded_2 = Reshape((int(dim/2),dim*2), input_shape=(dim*dim,1))(x2);
# 
#     model_full = Model([inputs_V,inputs_A], [decoded_1,decoded_2])
#     model_partial = Model([inputs_V,inputs_A], encoded)
#      
#     return (model_full, model_partial);

