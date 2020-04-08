def inceptionblockA(model,pool_features):
    #Branch 1
    branch1 = keras.layers.Conv2D(64,1, activation='relu')(model)
    
    #Branch 2
    branch2 = keras.layers.Conv2D(48,1, activation='relu')(model)
    branch2 =  keras.layers.ZeroPadding2D((2,2))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(64,5, activation='relu')(branch2)
    
    #Branch 3
    
    branch3 = keras.layers.Conv2D(64,1, activation='relu')(model)
    branch3 =  keras.layers.ZeroPadding2D((1,1))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(96,3, activation='relu')(branch3)
    branch3 =  keras.layers.ZeroPadding2D((1,1))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(96,3, activation='relu')(branch3)
    
    #Branch 4
    #We're Missing the Average Pool Layer Here For Now.
    #branch4 =  keras.layers.ZeroPadding2D((1,1))(model) # Adding 1x1 Padding
    branch4 = keras.layers.Conv2D(pool_features,1,activation='relu',strides=1)(model)
    
    
    block1_output = keras.layers.concatenate([branch1,branch2,branch3,branch4], axis=3)
    
    return block1_output

def inceptionblockB(model):
    #Branch 1
    branch1 = keras.layers.Conv2D(384,3, activation='relu',strides=2)(model) #3x3 Convolve.
    
    #Branch 2
    branch2 = keras.layers.Conv2D(64,1, activation='relu')(model)
    branch2 =  keras.layers.ZeroPadding2D((1,1))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(96,3, activation='relu')(branch2)
    branch2 = keras.layers.Conv2D(96,3, activation='relu',strides=2)(branch2)
    
    #Branch 3
    branch3 = keras.layers.MaxPooling2D(3, strides=2)(model) #MaxPool 3x3 Convolution
    
    
    blockB_output = keras.layers.concatenate([branch1,branch2,branch3], axis=3)
    
    return blockB_output
    

def inceptionblockC(model,channels_7x7):
    #Branch 1
    branch1 = keras.layers.Conv2D(192,1, activation='relu')(model) #3x3 Convolve.
    
    c7 = channels_7x7
    #Branch 2
    branch2 = keras.layers.Conv2D(c7,1, activation='relu')(model)
    branch2 =  keras.layers.ZeroPadding2D((0,3))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(c7,(1,7), activation='relu')(branch2)
    branch2 =  keras.layers.ZeroPadding2D((3,0))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(192,(7,1), activation='relu')(branch2)
    
    
    #Branch3 
    branch3 = keras.layers.Conv2D(c7,1, activation='relu')(model)
    
    branch3 =  keras.layers.ZeroPadding2D((3,0))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(c7,(7,1), activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((0,3))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(c7,(1,7), activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((3,0))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(c7,(7,1), activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((0,3))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(192,(1,7), activation='relu')(branch3)
    
    #Branch 4
#     branch4 =  keras.layers.ZeroPadding2D(1)(model) # Adding 1x1 Padding
#     branch4 = keras.layers.AveragePooling2D(3, strides=1)(branch4) #MaxPool 3x3 Convolution
    branch4 = keras.layers.Conv2D(192,1, activation='relu')(model)
    
    
    blockC_output = keras.layers.concatenate([branch1,branch2,branch3,branch4], axis=3)
    
    return blockC_output   
    
    
def inceptionblockD(model):
    #Branch 1
    branch1 = keras.layers.Conv2D(192,1, activation='relu')(model) #3x3 Convolve.
    branch1 = keras.layers.Conv2D(320,3,strides=2,activation='relu')(branch1)
    
    #Branch 2
    branch2 = keras.layers.Conv2D(192,1, activation='relu')(model)
    branch2 =  keras.layers.ZeroPadding2D((0,3))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(192,(1,7), activation='relu')(branch2)
    branch2 =  keras.layers.ZeroPadding2D((3,0))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(192,(7,1), activation='relu')(branch2)
    branch2 = keras.layers.Conv2D(192,3, activation='relu',strides=2)(branch2)
    
    
    
    #Branch3 

    branch3 = keras.layers.MaxPooling2D(3, strides=2)(model) #MaxPool 3x3 Convolution
    
    
    blockD_output = keras.layers.concatenate([branch1,branch2,branch3], axis=3)
    
    return blockD_output

    
def inceptionblockE(model):
    #Branch 1
    branch1 = keras.layers.Conv2D(320,1, activation='relu')(model) #3x3 Convolve.
    
    #Branch 2
    branch2 = keras.layers.Conv2D(384,1, activation='relu')(model)
    
    branch2 =  keras.layers.ZeroPadding2D((0,1))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(384,(1,3), activation='relu')(branch2)
    
    branch2 =  keras.layers.ZeroPadding2D((1,0))(branch2) # Adding 1x1 Padding
    branch2 = keras.layers.Conv2D(384,(3,1), activation='relu')(branch2)

    #Branch3 
    branch3 = keras.layers.Conv2D(448,1, activation='relu')(model)
    
    branch3 =  keras.layers.ZeroPadding2D(1)(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(384,3, activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((0,1))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(384,(1,3), activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((1,0))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(384,(3,1), activation='relu')(branch3)
    
    branch3 =  keras.layers.ZeroPadding2D((0,3))(branch3) # Adding 1x1 Padding
    branch3 = keras.layers.Conv2D(192,(1,7), activation='relu')(branch3)
    
    
    #Branch4 

    branch4 = keras.layers.Conv2D(192,1, activation='relu')(model)
    
    
    blockD_output = keras.layers.concatenate([branch1,branch2,branch3], axis=3)
    
    return blockD_output

def InceptionNetworkV3(input_image):
    First_layer = keras.Input(shape=input_image.shape) # Input Value
     
    first_conv = keras.layers.Conv2D(3,3, activation='relu',strides = 2)(First_layer) #3x3 Convolution
    second_conv =  keras.layers.Conv2D(32, 3, activation='relu')(first_conv) #3x3 Convolution
    add_padding_third_conv =  keras.layers.ZeroPadding2D((1,1))(second_conv) # Adding 1x1 Padding
    third_conv =  keras.layers.Conv2D(64, 3, activation='relu')(add_padding_third_conv) #3x3 Convolution
    maxpool_conv34 = keras.layers.MaxPooling2D(3, strides=2)(third_conv) #MaxPool 3x3 Convolution
    
    fourth_conv =  keras.layers.Conv2D(80, 1, activation='relu')(maxpool_conv34) #1x1 Convolution
    fifth_conv =  keras.layers.Conv2D(192,3, activation='relu')(fourth_conv) #1x1 Convolution
    
    maxpool_conv56 = keras.layers.MaxPooling2D(3, strides=2)(fifth_conv) #MaxPool 3x3 Convolution
    
    inception_block1 = inceptionblockA(maxpool_conv56,32)
    inception_block2 = inceptionblockA(inception_block1,64)
    inception_block3 = inceptionblockA(inception_block2,64)
    inception_block4 = inceptionblockB(inception_block3)
    inception_block5 = inceptionblockC(inception_block4,128)
    inception_block6 = inceptionblockC(inception_block5, 160)
    inception_block7 = inceptionblockC(inception_block6, 160)
    inception_block8 = inceptionblockC(inception_block7, 192)
    inception_block9 = inceptionblockD(inception_block8)
    inception_block10 = inceptionblockE(inception_block9)
    inception_block11 = inceptionblockE(inception_block10)
    
    
    #Add Auxillary branch Here.
    
    flatten =  keras.layers.Flatten()(inception_block11)
    dense_to_output = keras.layers.Dense(64, activation='relu')(flatten)
    output_classes = keras.layers.Dense(num_classes, activation='softmax')(dense_to_output)
    
    model = keras.Model(inputs=First_layer, outputs=output_classes)
    opt =RMSprop(lr=0.5)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
