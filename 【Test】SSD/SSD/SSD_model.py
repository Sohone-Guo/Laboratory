from keras.layers import *
from SSD.SSD_layer import *

def SSD300(input_shape,num_class):
    """ SSD300 architecture.
    
    - Arguments:
    
    - Return:
    
    - References:
        
    """
    
    net = {}
    
    
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1],input_shape[0])
    net["input"] = input_tensor
    net["conv1_1"] = Conv2D(kernel_size=[3,3],
                                   filters=64,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv1_1")(net["input"])
    
    net["conv1_2"] = Conv2D(kernel_size=[3,3],
                                   filters=64,
                                   activation="relu",
                                   padding="same",
                                   name="conv1_2")(net["conv1_1"])
    
    net["pool1"] = MaxPool2D(padding="same",
                             pool_size=[2,2],
                             strides=[2,2])(net["conv1_2"])
    
    
    # Block 2
    net["conv2_1"] = Conv2D(kernel_size=[3,3],
                                   filters=128,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv2_1")(net["pool1"])
    
    net["conv2_2"] = Conv2D(kernel_size=[3,3],
                                   filters=128,
                                   activation="relu",
                                   padding="same",
                                   name="conv2_2")(net["conv2_1"])
    
    net["pool2"] = MaxPool2D(padding="same",
                             pool_size=[2,2],
                             strides=[2,2])(net["conv2_2"])    
    
    
    # Block 3
    net["conv3_1"] = Conv2D(kernel_size=[3,3],
                                   filters=256,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv3_1")(net["pool2"])
    
    net["conv3_2"] = Conv2D(kernel_size=[3,3],
                                   filters=256,
                                   activation="relu",
                                   padding="same",
                                   name="conv3_2")(net["conv3_1"])
    
    net["conv3_3"] = Conv2D(kernel_size=[3,3],
                                   filters=256,
                                   activation="relu",
                                   padding="same",
                                   name="conv3_3")(net["conv3_2"])
    
    net["pool3"] = MaxPool2D(padding="same",
                             pool_size=[2,2],
                             strides=[2,2])(net["conv3_3"])    

    
    # Block 4
    net["conv4_1"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv4_1")(net["pool3"])
    
    net["conv4_2"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   activation="relu",
                                   padding="same",
                                   name="conv4_2")(net["conv4_1"])
    
    net["conv4_3"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   activation="relu",
                                   padding="same",
                                   name="conv4_3")(net["conv4_2"])
    
    net["pool4"] = MaxPool2D(padding="same",
                             pool_size=[2,2],
                             strides=[2,2])(net["conv4_3"])   
    

    # Block 5
    net["conv5_1"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv5_1")(net["pool4"])
    
    net["conv5_2"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   activation="relu",
                                   padding="same",
                                   name="conv5_2")(net["conv5_1"])
    
    net["conv5_3"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   activation="relu",
                                   padding="same",
                                   name="conv5_3")(net["conv5_2"])
    
    net["pool5"] = MaxPool2D(padding="same",
                             pool_size=[2,2],
                             strides=[2,2])(net["conv5_3"])
    
    
    # FC 6
    net["fc6"] = Conv2D(filters=1024,
                         kernel_size=[3,3],
                         dilation_rate=(6,6),
                         activation="relu",
                         padding="same",
                         name="fc6")(net["pool5"])
    
    
    # FC 7
    net["fc7"] = Conv2D(kernel_size=[1,1],
                               filters=1024,
                               activation="relu",
                               padding="same",
                               name="fc7")(net["fc6"])
    
    
    # Block 6
    net["conv6_1"] = Conv2D(kernel_size=[3,3],
                                   filters=256,
                                   strides=[1,1],
                                   activation="relu",
                                   padding="same",
                                   name="conv6_1")(net["fc7"])
    
    net["conv6_2"] = Conv2D(kernel_size=[3,3],
                                   filters=512,
                                   activation="relu",
                                   padding="same",
                                   name="conv6_2")(net["conv6_1"])
    
    
    # Block 7
    net["conv7_1"] = Conv2D(kernel_size=[1,1],
                                   filters=128,
                                   activation="relu",
                                   padding="same",
                                   name="conv7_1")(net["conv6_2"])   
    
    net["conv7_2"] = ZeroPadding2D()(net["conv7_1"])
    net["conv7_2"] = Conv2D(kernel_size=[3,3],
                                   filters=256,
                                   subsample=(2,2),
                                   activation="relu",
                                   padding="valid",
                                   name="conv7_2")(net["conv7_2"])
    
    
    # Block 8
    net["conv8_1"] = Conv2D(kernel_size=[1,1],
                                   filters=128,
                                   subsample=(2,2),
                                   activation="relu",
                                   padding="same",
                                   name="conv8_1")(net["conv7_2"]) 
    
    net["conv8_2"] = Conv2D(kernel_size=[3,3],
                               filters=256,
                               subsample=(2,2),
                               activation="relu",
                               padding="same",
                               name="conv8_2")(net["conv8_1"]) 
 

    # Last Pool
    net["pool6"] = GlobalAveragePooling2D(name="pool6")(net["conv8_2"])
    
    
    # Prediction from conv4_3
    # conv4_3 mbox
    net["conv4_3_norm"] = Normalize(20, name="conv4_3_norm")(net["conv4_3"])
    num_priors = 3
    net["conv4_3_norm_mbox_loc"] = Conv2D(kernel_size=[3,3],
                                           filters=num_priors*4,
                                           padding="same",
                                           activation="relu",
                                           name="conv4_3_norm_mbox_loc")(net["conv4_3_norm"])
    flatten = Flatten(name="conv4_3_norm_mbox_loc_flat")
    net["conv4_3_norm_mbox_loc_flat"] = flatten(net["conv4_3_norm_mbox_loc"])
    
    # conv4_3 conf
    name = "conv4_3_norm_mbox_conf_{}".format(num_class)
    net["conv4_3_norm_mbox_conf"] = Conv2D(kernel_size=[3,3],
                                           filters=num_priors*num_class,
                                           padding="same",
                                           name=name)(net["conv4_3_norm"])
    
    flatten = Flatten(name="conv4_3_norm_mbox_conf_flat")
    net["conv4_3_norm_mbox_conf_flat"] = flatten(net["conv4_3_norm_mbox_conf"])
    
    # conv4_3 priorbox
    priorbox = PriorBox(img_size,30.0,aspect_ratios=[2],
                        variances=[0.1,0.1,0.2,0.2],
                        name="conv4_3_norm_mbox_priorbox")
    net["conv4_3_norm_mbox_priorbox"] = priorbox(net["conv4_3_norm"])
    print("The loc:",net["conv4_3_norm_mbox_loc"])
    print("The conf: ",net["conv4_3_norm_mbox_conf"])
    print("This prior size is",net["conv4_3_norm_mbox_priorbox"])
    
    