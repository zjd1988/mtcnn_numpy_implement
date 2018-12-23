#-*- coding:utf-8 -*-
import numpy as np
import caffe
import cv2
import pickle
import NumPyCNN as numpycnn
import time


def check_data(ref_data, com_data):
    check_result = ref_data - com_data
    check_sum = np.sum(np.abs(check_result))
    return check_sum

def custom_prelu(bottom_data, weights):
    #if len(bottom_data.shape) != 3:
        #print("wrong shape !!!!")

    for index in range(bottom_data.shape[0]):
        #temp = bottom_data[index, :, :]
        temp = bottom_data[index]
        temp[temp < 0] = temp[temp < 0] * weights[index]
        #bottom_data[index, :, :] = temp
        bottom_data[index] = temp

    return bottom_data
    

def custom_conv2d(bottom_data, weights, bias):
    input = bottom_data.copy()
    shape = bottom_data.shape
    if len(shape) > 3:
        input = input.reshape((shape[1], shape[2], shape[3]))

    input = input.transpose((1, 2, 0))
    #weights = weights.transpose((0, 1, 2, 3))  wrong
    #weights = weights.transpose((0, 1, 3, 2))  wrong
    weights = weights.transpose((0, 2, 3, 1))
    #weights = weights.transpose((0, 2, 1, 3))   wrong
    #weights = weights.transpose((0, 3, 1, 2))
    #weights = weights.transpose((0, 3, 2, 1))

    conv_result = numpycnn.conv(input, weights)
    for bias_index in range(len(bias)):
        conv_result[:, :, bias_index] = conv_result[:, :, bias_index] + bias[bias_index]
    return conv_result

def custom_pool(bottom_data, size, stride):
    #return numpycnn.pooling(bottom_data, size, stride)
    return numpycnn.pooling_new(bottom_data, size, stride)

def custom_softmax(bottom_data):
    if len(bottom_data.shape) == 4:
        batch_num = bottom_data.shape[0]
        channel_num = bottom_data.shape[1]
        height = bottom_data.shape[2]
        width = bottom_data.shape[3]
        sum_bottom = np.empty((height, width))
        softmax_result = np.empty((batch_num, channel_num, height, width))
        for height_index in range(height):
            for width_index in range(width):
                sum_bottom[height_index][width_index] = np.sum(np.exp(bottom_data[:, :, height_index, width_index]))

            
        for height_index in range(height):
            for width_index in range(width):
                softmax_result[:, :, height_index, width_index] = np.exp(bottom_data[:, :, height_index, width_index]) / sum_bottom[height_index][width_index]
        
    return softmax_result
                        

def custom_softmax_bak(bottom_data, axis = 1):
    if axis > (len(bottom_data.shape) - 1):
        return -1
    if len(bottom_data.shape) == 1:
        sum_bottom = np.empty(1)
        softmax_result = np.empty(bottom_data.shape)
        sum_bottom = np.sum(np.exp(bottom_data[:]))
        softmax_result[:] = np.exp(bottom_data[:]) / sum_bottom
    elif len(bottom_data.shape) == 2:
        dim0 = bottom_data.shape[0]
        dim1 = bottom_data.shape[1]
        if axis == 0:
            sum_bottom = np.empty(dim1)
            softmax_result = np.empty(bottom_data.shape)
            for index1 in range(dim1):
                sum_bottom[index1] = np.sum(np.exp(bottom_data[:, index1]))
                softmax_result[:, index1] = np.exp(bottom_data[:, index1]) / sum_bottom[index1]

        else:
            sum_bottom = np.empty(dim0)
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                sum_bottom[index0] = np.sum(np.exp(bottom_data[index0, :]))
                softmax_result[index0, :] = np.exp(bottom_data[index0, :]) / sum_bottom[index0]
                
    elif len(bottom_data.shape) == 3:
        dim0 = bottom_data.shape[0]
        dim1 = bottom_data.shape[1]
        dim2 = bottom_data.shape[2]
        if axis == 0:
            sum_bottom = np.empty((dim1, dim2))
            softmax_result = np.empty(bottom_data.shape)
            for index1 in range(dim1):
                for index2 in range(dim2):
                    sum_bottom[index1][index2] = np.sum(np.exp(bottom_data[:, index1, index2]))
                    softmax_result[:, index1, index2] = np.exp(bottom_data[:, index1, index2]) / sum_bottom[index1][index2]
                    
        elif axis == 1:
            sum_bottom = np.empty((dim0, dim2))
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                for index2 in range(dim2):
                    sum_bottom[index0][index2] = np.sum(np.exp(bottom_data[index0, :, index2]))
                    softmax_result[index0, :, index2] = np.exp(bottom_data[index0, :, index2]) / sum_bottom[index0][index2]
                    
        else:
            sum_bottom = np.empty((dim0, dim1))
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                for index1 in range(dim1):
                    sum_bottom[index0][index1] = np.sum(np.exp(bottom_data[index0, index1, :]))
                    softmax_result[index0, index1, :] = np.exp(bottom_data[index0, index1, :]) / sum_bottom[index0][index1]

    elif len(bottom_data.shape) == 4:
        dim0 = bottom_data.shape[0]
        dim1 = bottom_data.shape[1]
        dim2 = bottom_data.shape[2]
        dim3 = bottom_data.shape[3]
        if axis == 0:
            sum_bottom = np.empty((dim1, dim2, dim3))
            softmax_result = np.empty(bottom_data.shape)
            for index1 in range(dim1):
                for index2 in range(dim2):
                    for index3 in range(dim3):
                        sum_bottom[index1][index2][index3] = np.sum(np.exp(bottom_data[:, index1, index2, index3]))
                        softmax_result[:, index1, index2, index3] = np.exp(bottom_data[:, index1, index2, index3]) / sum_bottom[index1][index2][index3]
                        
        elif axis == 1:
            sum_bottom = np.empty((dim0, dim2, dim3))
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                for index2 in range(dim2):
                    for index3 in range(dim3):
                        sum_bottom[index0][index2][index3] = np.sum(np.exp(bottom_data[index0, :, index2, index3]))
                        softmax_result[index0, :, index2, index3] = np.exp(bottom_data[index0, :, index2, index3]) / sum_bottom[index0][index2][index3]
                        
        elif axis == 2:
            sum_bottom = np.empty((dim0, dim1, dim3))
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                for index1 in range(dim1):
                    for index3 in range(dim3):
                        sum_bottom[index0][index1][index3] = np.sum(np.exp(bottom_data[index0, index1, :, index3]))
                        softmax_result[index0, index1, :, index3] = np.exp(bottom_data[index0, index1, :, index3]) / sum_bottom[index0][index1][index3]

        else:
            sum_bottom = np.empty((dim0, dim1, dim2))
            softmax_result = np.empty(bottom_data.shape)
            for index0 in range(dim0):
                for index1 in range(dim1):
                    for index2 in range(dim2):                    
                        sum_bottom[index0][index1][index2] = np.sum(np.exp(bottom_data[index0, index1, index2, :]))
                        softmax_result[index0, index1, index2, :] = np.exp(bottom_data[:, index1, index2, index3]) / sum_bottom[index0][index1][index2]
                       
    else:
        return -1
    return softmax_result                        

def compare_PNet_blob_with_numpy(net):
    net_name = "PNet"
    print("{} is run....".format(net_name))
    f = open("PNet_weights.pkl", "rb")
    PNet_para = pickle.load(f)
    f.close()
    bottom = "data"
    relu = "PReLU1"

    layer_name = "conv1"
    bottom_data = net.blobs[bottom].data
    top_data = net.blobs[layer_name].data
    conv1 = []
    weights = PNet_para[1]["weights"][0]
    bias = PNet_para[1]["weights"][1]
    prelu_weights = PNet_para[2]["weights"][0]
    conv1 = custom_conv2d(bottom_data, weights, bias)
    conv1 = conv1.transpose(2, 0, 1)
    conv1 = custom_prelu(conv1, prelu_weights)
    conv1 = conv1.reshape((1, conv1.shape[0], conv1.shape[1], conv1.shape[2]))
    check_result = check_data(top_data, conv1)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "pool1"
    pool1 = []
    conv1 = conv1.reshape((conv1.shape[1], conv1.shape[2], conv1.shape[3]))
    conv1 = conv1.transpose(1, 2, 0)
    pool1 = custom_pool(conv1, 2, 2)
    top_data = net.blobs[layer_name].data
    pool1 = pool1.transpose(2, 0, 1)
    pool1 = pool1.reshape((1, pool1.shape[0], pool1.shape[1], pool1.shape[2]))
    check_result = check_data(top_data, pool1)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "conv2"
    bottom_data = pool1
    top_data = net.blobs[layer_name].data
    conv2 = []
    weights = PNet_para[4]["weights"][0]
    bias = PNet_para[4]["weights"][1]
    prelu_weights = PNet_para[5]["weights"][0]
    conv2 = custom_conv2d(bottom_data, weights, bias)
    conv2 = conv2.transpose(2, 0, 1)
    conv2 = custom_prelu(conv2, prelu_weights)
    conv2 = conv2.reshape((1, conv2.shape[0], conv2.shape[1], conv2.shape[2]))
    check_result = check_data(top_data, conv2)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "conv3"
    bottom_data = conv2
    top_data = net.blobs[layer_name].data
    conv3 = []
    weights = PNet_para[6]["weights"][0]
    bias = PNet_para[6]["weights"][1]
    prelu_weights = PNet_para[7]["weights"][0]
    conv3 = custom_conv2d(bottom_data, weights, bias)
    conv3 = conv3.transpose(2, 0, 1)
    conv3 = custom_prelu(conv3, prelu_weights)
    conv3 = conv3.reshape((1, conv3.shape[0], conv3.shape[1], conv3.shape[2]))
    check_result = check_data(top_data, conv3)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "conv4-1"
    bottom_data = conv3
    top_data = net.blobs[layer_name].data
    conv4_1 = []
    weights = PNet_para[9]["weights"][0]
    bias = PNet_para[9]["weights"][1]
    conv4_1 = custom_conv2d(bottom_data, weights, bias)
    conv4_1 = conv4_1.transpose(2, 0, 1)
    conv4_1 = conv4_1.reshape((1, conv4_1.shape[0], conv4_1.shape[1], conv4_1.shape[2]))
    check_result = check_data(top_data, conv4_1)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "conv4-2"
    bottom_data = conv3
    top_data = net.blobs[layer_name].data
    conv4_2 = []
    weights = PNet_para[10]["weights"][0]
    bias = PNet_para[10]["weights"][1]
    conv4_2 = custom_conv2d(bottom_data, weights, bias)
    conv4_2 = conv4_2.transpose(2, 0, 1)
    conv4_2 = conv4_2.reshape((1, conv4_2.shape[0], conv4_2.shape[1], conv4_2.shape[2]))
    check_result = check_data(top_data, conv4_2)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "prob1"
    bottom_data = conv4_1
    top_data = net.blobs[layer_name].data
    prob1 = []
    #prob1 = custom_softmax(bottom_data)
    prob1 = custom_softmax_bak(bottom_data, 1)
    check_result = check_data(top_data, prob1)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    out = {}
    out["prob1"] = prob1
    out["conv4-2"] = conv4_2
    return out

def compare_RNet_blob_with_numpy(net):
    net_name = "RNet"
    print("{} is run....".format(net_name))    
    f = open("RNet_weights.pkl", "rb")
    RNet_para = pickle.load(f)
    f.close()
    bottom = "data"
    prelu1 = "prelu1"
    layer_name = "conv1"

    bottom_data = net.blobs[bottom].data
    top_data = net.blobs[layer_name].data
    conv1 = np.empty(top_data.shape)
    weights = RNet_para[1]["weights"][0]
    bias = RNet_para[1]["weights"][1]
    prelu_weights = RNet_para[2]["weights"][0]
    for index in range(bottom_data.shape[0]):
        temp_conv1 = custom_conv2d(bottom_data[index], weights, bias)
        temp_conv1 = temp_conv1.transpose(2, 0, 1)
        temp_conv1 = custom_prelu(temp_conv1, prelu_weights)
        temp_conv1 = temp_conv1.reshape((1, temp_conv1.shape[0], temp_conv1.shape[1], temp_conv1.shape[2]))
        conv1[index] = temp_conv1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv1)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "pool1"
    top_data = net.blobs[layer_name].data
    pool1 = np.empty(top_data.shape)
    for index in range(conv1.shape[0]):
        temp_conv1 = conv1[index]
        temp_conv1 = temp_conv1.transpose(1, 2, 0)
        temp_pool1 = custom_pool(temp_conv1, 3, 2)
        
        temp_pool1 = temp_pool1.transpose(2, 0, 1)
        temp_pool1 = temp_pool1.reshape((1, temp_pool1.shape[0], temp_pool1.shape[1], temp_pool1.shape[2]))
        pool1[index] = temp_pool1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, pool1)
    print("check layer {}, and result is {}".format(layer_name, check_result))   


    layer_name = "conv2"
    top_data = net.blobs[layer_name].data
    conv2 = np.empty(top_data.shape)
    weights = RNet_para[4]["weights"][0]
    bias = RNet_para[4]["weights"][1]
    prelu_weights = RNet_para[5]["weights"][0]
    for index in range(pool1.shape[0]):
        temp_conv2 = custom_conv2d(pool1[index], weights, bias)
        temp_conv2 = temp_conv2.transpose(2, 0, 1)
        temp_conv2 = custom_prelu(temp_conv2, prelu_weights)
        temp_conv2 = temp_conv2.reshape((1, temp_conv2.shape[0], temp_conv2.shape[1], temp_conv2.shape[2]))
        conv2[index] = temp_conv2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv2)
    print("check layer {}, and result is {}".format(layer_name, check_result))   

    layer_name = "pool2"
    top_data = net.blobs[layer_name].data
    pool2 = np.empty(top_data.shape)
    for index in range(conv2.shape[0]):
        temp_conv2 = conv2[index]
        temp_conv2 = temp_conv2.transpose(1, 2, 0)
        temp_pool2 = custom_pool(temp_conv2, 3, 2)
        
        temp_pool2 = temp_pool2.transpose(2, 0, 1)
        temp_pool2 = temp_pool2.reshape((1, temp_pool2.shape[0], temp_pool2.shape[1], temp_pool2.shape[2]))
        pool2[index] = temp_pool2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, pool2)
    print("check layer {}, and result is {}".format(layer_name, check_result))  


    layer_name = "conv3"
    top_data = net.blobs[layer_name].data
    conv3 = np.empty(top_data.shape)
    weights = RNet_para[7]["weights"][0]
    bias = RNet_para[7]["weights"][1]
    prelu_weights = RNet_para[8]["weights"][0]
    for index in range(pool2.shape[0]):
        temp_conv3 = custom_conv2d(pool2[index], weights, bias)
        temp_conv3 = temp_conv3.transpose(2, 0, 1)
        temp_conv3 = custom_prelu(temp_conv3, prelu_weights)
        temp_conv3 = temp_conv3.reshape((1, temp_conv3.shape[0], temp_conv3.shape[1], temp_conv3.shape[2]))
        conv3[index] = temp_conv3
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv3)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "conv4"
    top_data = net.blobs[layer_name].data
    conv4 = np.empty(top_data.shape)
    weights = RNet_para[9]["weights"][0]
    bias = RNet_para[9]["weights"][1]
    prelu_weights = RNet_para[10]["weights"][0]
    for index in range(conv3.shape[0]):
        temp_conv4 = conv3[index]
        temp_conv4 = temp_conv4.reshape(1, -1)
        temp_conv4 = np.matmul(temp_conv4, weights.T)
        temp_conv4 = temp_conv4 + bias.reshape(1, 128)
        temp_conv4 = temp_conv4.reshape(128, 1)
        temp_conv4 = custom_prelu(temp_conv4, prelu_weights.reshape(128, 1))
        temp_conv4 = temp_conv4.reshape(1, 128)

        conv4[index] = temp_conv4
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv4)
    print("check layer {}, and result is {}".format(layer_name, check_result)) 


    layer_name = "conv5-1"
    top_data = net.blobs[layer_name].data
    conv5_1 = np.empty(top_data.shape)
    weights = RNet_para[12]["weights"][0]
    bias = RNet_para[12]["weights"][1]
    for index in range(conv4.shape[0]):
        temp_conv5_1 = conv4[index]
        temp_conv5_1 = temp_conv5_1.reshape(1, -1)
        temp_conv5_1 = np.matmul(temp_conv5_1, weights.T)
        temp_conv5_1 = temp_conv5_1 + bias.reshape(-1, 2)
        conv5_1[index] = temp_conv5_1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv5_1)
    print("check layer {}, and result is {}".format(layer_name, check_result)) 

    layer_name = "conv5-2"
    top_data = net.blobs[layer_name].data
    conv5_2 = np.empty(top_data.shape)
    weights = RNet_para[13]["weights"][0]
    bias = RNet_para[13]["weights"][1]
    for index in range(conv4.shape[0]):
        temp_conv5_2 = conv4[index]
        temp_conv5_2 = temp_conv5_2.reshape(1, -1)
        temp_conv5_2 = np.matmul(temp_conv5_2, weights.T)
        temp_conv5_2 = temp_conv5_2 + bias.reshape(-1, 4)
        conv5_2[index] = temp_conv5_2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv5_2)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "prob1"
    bottom_data = conv5_1
    top_data = net.blobs[layer_name].data
    prob1 = []
    #prob1 = custom_softmax(bottom_data)
    prob1 = custom_softmax_bak(bottom_data, 1)
    check_result = check_data(top_data, prob1)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    out= {}
    out["prob1"] = prob1
    out["conv5-2"] = conv5_2
    return out

def compare_ONet_blob_with_numpy(net):
    net_name = "ONet"
    print("{} is run....".format(net_name))    
    f = open("ONet_weights.pkl", "rb")
    ONet_para = pickle.load(f)
    f.close()
    bottom = "data"
    layer_name = "conv1"

    bottom_data = net.blobs[bottom].data
    top_data = net.blobs[layer_name].data
    conv1 = np.empty(top_data.shape)
    weights = ONet_para[1]["weights"][0]
    bias = ONet_para[1]["weights"][1]
    prelu_weights = ONet_para[2]["weights"][0]
    for index in range(bottom_data.shape[0]):
        temp_conv1 = custom_conv2d(bottom_data[index], weights, bias)
        temp_conv1 = temp_conv1.transpose(2, 0, 1)
        temp_conv1 = custom_prelu(temp_conv1, prelu_weights)
        temp_conv1 = temp_conv1.reshape((1, temp_conv1.shape[0], temp_conv1.shape[1], temp_conv1.shape[2]))
        conv1[index] = temp_conv1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv1)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "pool1"
    top_data = net.blobs[layer_name].data
    pool1 = np.empty(top_data.shape)
    for index in range(conv1.shape[0]):
        temp_conv1 = conv1[index]
        temp_conv1 = temp_conv1.transpose(1, 2, 0)
        temp_pool1 = custom_pool(temp_conv1, 3, 2)
        
        temp_pool1 = temp_pool1.transpose(2, 0, 1)
        temp_pool1 = temp_pool1.reshape((1, temp_pool1.shape[0], temp_pool1.shape[1], temp_pool1.shape[2]))
        pool1[index] = temp_pool1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, pool1)
    print("check layer {}, and result is {}".format(layer_name, check_result))   



    layer_name = "conv2"
    top_data = net.blobs[layer_name].data
    conv2 = np.empty(top_data.shape)
    weights = ONet_para[4]["weights"][0]
    bias = ONet_para[4]["weights"][1]
    prelu_weights = ONet_para[5]["weights"][0]
    for index in range(pool1.shape[0]):
        temp_conv2 = custom_conv2d(pool1[index], weights, bias)
        temp_conv2 = temp_conv2.transpose(2, 0, 1)
        temp_conv2 = custom_prelu(temp_conv2, prelu_weights)
        temp_conv2 = temp_conv2.reshape((1, temp_conv2.shape[0], temp_conv2.shape[1], temp_conv2.shape[2]))
        conv2[index] = temp_conv2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv2)
    print("check layer {}, and result is {}".format(layer_name, check_result))   

    layer_name = "pool2"
    top_data = net.blobs[layer_name].data
    pool2 = np.empty(top_data.shape)
    for index in range(conv2.shape[0]):
        temp_conv2 = conv2[index]
        temp_conv2 = temp_conv2.transpose(1, 2, 0)
        temp_pool2 = custom_pool(temp_conv2, 3, 2)
        
        temp_pool2 = temp_pool2.transpose(2, 0, 1)
        temp_pool2 = temp_pool2.reshape((1, temp_pool2.shape[0], temp_pool2.shape[1], temp_pool2.shape[2]))
        pool2[index] = temp_pool2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, pool2)
    print("check layer {}, and result is {}".format(layer_name, check_result))  



    layer_name = "conv3"
    top_data = net.blobs[layer_name].data
    conv3 = np.empty(top_data.shape)
    weights = ONet_para[7]["weights"][0]
    bias = ONet_para[7]["weights"][1]
    prelu_weights = ONet_para[8]["weights"][0]
    for index in range(pool2.shape[0]):
        temp_conv3 = custom_conv2d(pool2[index], weights, bias)
        temp_conv3 = temp_conv3.transpose(2, 0, 1)
        temp_conv3 = custom_prelu(temp_conv3, prelu_weights)
        temp_conv3 = temp_conv3.reshape((1, temp_conv3.shape[0], temp_conv3.shape[1], temp_conv3.shape[2]))
        conv3[index] = temp_conv3
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv3)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "pool3"
    top_data = net.blobs[layer_name].data
    pool3 = np.empty(top_data.shape)
    for index in range(conv3.shape[0]):
        temp_conv3 = conv3[index]
        temp_conv3 = temp_conv3.transpose(1, 2, 0)
        temp_pool3 = custom_pool(temp_conv3, 2, 2)
        
        temp_pool3 = temp_pool3.transpose(2, 0, 1)
        temp_pool3 = temp_pool3.reshape((1, temp_pool3.shape[0], temp_pool3.shape[1], temp_pool3.shape[2]))
        pool3[index] = temp_pool3
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, pool3)
    print("check layer {}, and result is {}".format(layer_name, check_result))  


    layer_name = "conv4"
    top_data = net.blobs[layer_name].data
    conv4 = np.empty(top_data.shape)
    weights = ONet_para[10]["weights"][0]
    bias = ONet_para[10]["weights"][1]
    prelu_weights = ONet_para[11]["weights"][0]
    for index in range(pool3.shape[0]):
        temp_conv4 = custom_conv2d(pool3[index], weights, bias)
        temp_conv4 = temp_conv4.transpose(2, 0, 1)
        temp_conv4 = custom_prelu(temp_conv4, prelu_weights)
        temp_conv4 = temp_conv4.reshape((1, temp_conv4.shape[0], temp_conv4.shape[1], temp_conv4.shape[2]))
        conv4[index] = temp_conv4
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv4)
    print("check layer {}, and result is {}".format(layer_name, check_result)) 



    layer_name = "conv5"
    top_data = net.blobs[layer_name].data
    conv5 = np.empty(top_data.shape)
    weights = ONet_para[12]["weights"][0]
    bias = ONet_para[12]["weights"][1]
    prelu_weights = ONet_para[14]["weights"][0]
    for index in range(conv4.shape[0]):
        temp_conv5 = conv4[index]
        temp_conv5 = temp_conv5.reshape(1, -1)
        temp_conv5 = np.matmul(temp_conv5, weights.T)
        temp_conv5 = temp_conv5 + bias.reshape(-1, 256)
        temp_conv5 = temp_conv5.reshape(256, 1)
        temp_conv5 = custom_prelu(temp_conv5, prelu_weights.reshape(256, 1))
        temp_conv5 = temp_conv5.reshape(1, -1)
        conv5[index] = temp_conv5
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv5)
    print("check layer {}, and result is {}".format(layer_name, check_result)) 

    
    layer_name = "conv6-1"
    top_data = net.blobs[layer_name].data
    conv6_1 = np.empty(top_data.shape)
    weights = ONet_para[16]["weights"][0]
    bias = ONet_para[16]["weights"][1]
    for index in range(conv5.shape[0]):
        temp_conv6_1 = conv5[index]
        temp_conv6_1 = temp_conv6_1.reshape(1, -1)
        temp_conv6_1 = np.matmul(temp_conv6_1, weights.T)
        temp_conv6_1 = temp_conv6_1 + bias.reshape(-1, 2)
        conv6_1[index] = temp_conv6_1
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv6_1)
    print("check layer {}, and result is {}".format(layer_name, check_result)) 

    layer_name = "conv6-2"
    top_data = net.blobs[layer_name].data
    conv6_2 = np.empty(top_data.shape)
    weights = ONet_para[17]["weights"][0]
    bias = ONet_para[17]["weights"][1]
    for index in range(conv5.shape[0]):
        temp_conv6_2 = conv5[index]
        temp_conv6_2 = temp_conv6_2.reshape(1, -1)
        temp_conv6_2 = np.matmul(temp_conv6_2, weights.T)
        temp_conv6_2 = temp_conv6_2 + bias.reshape(-1, 4)
        conv6_2[index] = temp_conv6_2
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv6_2)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    layer_name = "conv6-3"
    top_data = net.blobs[layer_name].data
    conv6_3 = np.empty(top_data.shape)
    weights = ONet_para[18]["weights"][0]
    bias = ONet_para[18]["weights"][1]
    for index in range(conv5.shape[0]):
        temp_conv6_3 = conv5[index]
        temp_conv6_3 = temp_conv6_3.reshape(1, -1)
        temp_conv6_3 = np.matmul(temp_conv6_3, weights.T)
        temp_conv6_3 = temp_conv6_3 + bias.reshape(-1, 10)
        conv6_3[index] = temp_conv6_3
        #print("layer name {} index {}".format(layer_name, index))
    check_result = check_data(top_data, conv6_3)
    print("check layer {}, and result is {}".format(layer_name, check_result))


    layer_name = "prob1"
    bottom_data = conv6_1
    top_data = net.blobs[layer_name].data
    prob1 = []
    #prob1 = custom_softmax(bottom_data)
    prob1 = custom_softmax_bak(bottom_data, 1)
    check_result = check_data(top_data, prob1)
    print("check layer {}, and result is {}".format(layer_name, check_result))

    out= {}
    out["prob1"] = prob1
    out["conv6-2"] = conv6_2
    out["conv6-3"] = conv6_3
    return out

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print('tmph', tmph)
    #print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    #print("dy"  ,dy )
    #print("dx"  ,dx )
    #print("y "  ,y )
    #print("x "  ,x )
    #print("edy" ,edy)
    #print("edx" ,edx)
    #print("ey"  ,ey )
    #print("ex"  ,ex )


    #print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    #print("dx1.shape", dx1.shape)
    #print('map.shape', map.shape)
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)

    return boundingbox_out.T


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()
    
    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    

    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print("scale:", scale)


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        ########################################
        ########################################
        ########################################
        out = compare_PNet_blob_with_numpy(PNet)
        ########################################
        ########################################
        ########################################
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            #print(boxes[4:9])
            #print('im_data', im_data[0:5, 0:5, 0], '\n')
            #print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    print("[1]:",total_boxes.shape[0])
    #print(total_boxes)
    #return total_boxes, [] 
    if RNet == None:
        return None, None

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        #print("[3]:",total_boxes.shape[0])
        #print(regh)
        #print(regw)
        #print('t1',t1)
        #print(total_boxes)

        total_boxes = rerec(total_boxes) # convert box to square
        print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        print("[4.5]:",total_boxes.shape[0])
        #print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print(total_boxes.shape)
    #print(total_boxes)



    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print('tmph', tmph)
        #print('tmpw', tmpw)
        #print("y,ey,x,ex", y, ey, x, ex, )
        #print("edy", edy)

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
          
            #print("dx[k], edx[k]:", dx[k], edx[k])
            #print("dy[k], edy[k]:", dy[k], edy[k])
            #print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            #print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            #print("tmp", tmp.shape)
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print('tempimg', tempimg[k,:,:,:].shape)
            #print(tempimg[k,0:5,0:5,0] )
            #print(tempimg[k,0:5,0:5,1] )
            #print(tempimg[k,0:5,0:5,2] )
            #print(k)
    
        #print(tempimg.shape)
        #print(tempimg[0,0,0,:])
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print(tempimg[0,:,0,0])
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()
        
        ############################
        ############################
        #############################
        out = compare_RNet_blob_with_numpy(RNet)
        ############################
        ############################
        #############################        
        #print(out['conv5-2'].shape)
        #print(out['prob1'].shape)

        score = out['prob1'][:,1]
        #print('score', score)
        pass_t = np.where(score>threshold[1])[0]
        #print('pass_t', pass_t)
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        print("[5]:",total_boxes.shape[0])
        #print(total_boxes)

        #print("1.5:",total_boxes.shape)
        
        mv = out['conv5-2'][pass_t, :].T
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        print("2:",total_boxes.shape)
        if ONet == None:
            return None, None

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print('tmpw', tmpw)
            #print('tmph', tmph)
            #print('y ', y)
            #print('ey', ey)
            #print('x ', x)
            #print('ex', ex)
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()

            ############################
            ############################
            #############################
            out = compare_ONet_blob_with_numpy(ONet)
            ############################
            ############################
            #############################                  
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    print("3:",total_boxes.shape)

    return total_boxes, points

def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

  


def tic():
    globals()['tt'] = time.clock()

def toc():
    print('\nElapsed time: {} seconds\n'.format(time.clock()-globals()['tt']))

def main():

    ##1、load caffe model and set model parameter
    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)

    ##2、load test image,and backup
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bak = img.copy()

    tic()
    boundingboxes, points = detect_face(img, minsize, PNet, RNet, ONet, threshold, False, factor)
    toc()

    img_bak = drawBoxes(img_bak, boundingboxes)
    cv2.imshow('img', img_bak)
    ch = cv2.waitKey(0) & 0xFF
    if ch == 27:
        pass

if __name__ == "__main__":
    main()