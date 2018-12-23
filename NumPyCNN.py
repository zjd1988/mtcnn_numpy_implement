import numpy
import sys

"""
Convolutional neural network implementation using NumPy.
An article describing this project is titled "Building Convolutional Neural Network using NumPy from Scratch". It is available in these links: https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad/
https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741

The project is tested using Python 3.5.2 installed inside Anaconda 4.2.0 (64-bit)
NumPy version used is 1.14.0

For more info., contact me:
    Ahmed Fawzy Gad
    KDnuggets: https://www.kdnuggets.com/author/ahmed-gad
    LinkedIn: https://www.linkedin.com/in/ahmedfgad
    Facebook: https://www.facebook.com/ahmed.f.gadd
    ahmed.f.gad@gmail.com
    ahmed.fawzy@ci.menofia.edu.eg
"""

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in numpy.uint16(numpy.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in numpy.uint16(numpy.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)):r+numpy.uint16(numpy.ceil(filter_size/2.0)), 
                              c-numpy.uint16(numpy.floor(filter_size/2.0)):c+numpy.uint16(numpy.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    if filter_size % 2 == 0:
        final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0) + 1, 
                              numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0) + 1]
    else:
        final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0), 
                              numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]        
    return final_result
def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    
    #if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
    #    print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
    #    sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        #print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.
    

def pooling(feature_map, size=2, stride=2):
    if len(feature_map.shape) != 3:
        return -1

    feature_map_bak = feature_map.copy()
    if feature_map.shape[0] % 2 == 1:
        dim0 = feature_map.shape[0] + 1
    else:
        dim0 = feature_map.shape[0]

    if feature_map.shape[1] % 2 == 1:
        dim1 = feature_map.shape[1] + 1
    else:
        dim1 = feature_map.shape[1]
    
    feature_map_new = numpy.empty((dim0, dim1, feature_map.shape[2]))
    feature_map_new[0:feature_map.shape[0], 0:feature_map.shape[1], :] = feature_map[:, :, :]
    feature_map_new[feature_map.shape[0]:, :, :] = feature_map.min()
    feature_map_new[:, feature_map.shape[1]:, :] = feature_map.min()
    feature_map = feature_map_new
    #Preparing the output of the pooling operation.
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride+1),
                            numpy.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1

    feature_map = feature_map_bak
    return pool_out


def pooling_new(feature_map, size=2, stride=2):
    if len(feature_map.shape) != 3:
        return -1

    feature_map_bak = feature_map.copy()
    dim0 = feature_map.shape[0]
    dim1 = feature_map.shape[1]
    if stride >= size:
        if dim0 % stride != 0:
            dim0 = dim0 + stride - dim0 % stride
        if dim1 % stride != 0:
            dim1 = dim1 + stride - dim1 % stride
    else:
        if dim0 % size != 0:
            dim0 = dim0 + size - dim0 % size
        if dim1 % size != 0:
            dim1 = dim1 + size - dim1 % size
    
    feature_map_new = numpy.empty((dim0, dim1, feature_map.shape[2]))
    feature_map_new[0:feature_map.shape[0], 0:feature_map.shape[1], :] = feature_map[:, :, :]
    feature_map_new[feature_map.shape[0]:, :, :] = feature_map.min()
    feature_map_new[:, feature_map.shape[1]:, :] = feature_map.min()
    feature_map = feature_map_new
    #Preparing the output of the pooling operation.
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride+1),
                            numpy.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1

    feature_map = feature_map_bak
    dim0 = feature_map.shape[0]
    dim1 = feature_map.shape[1]
    if dim0 % stride != 0 and dim0 % stride >= size / 2:
        dim0 = int(numpy.floor(dim0 / stride) + 1)
    else:
        dim0 = int(dim0 / stride)

    if dim1 % stride != 0 and dim1 % stride >= size / 2:
        dim1 = int(numpy.floor(dim1 / stride) + 1)
    else:
        dim1 = int(dim1 / stride)

    pool_out = pool_out[0:dim0, 0:dim1, :]
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out
