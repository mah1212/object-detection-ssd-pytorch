# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:14:56 2018


"""
# =============================================================================
# 
# Object detection using pytorch implementation of SSD
# 
# =============================================================================

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio



# Detector class
def detect(frame, net, transform):
    
    # get height, width from frame
    # height, width = frame.shape[0, 1] or below code
    height, width = frame.shape[:2] # take 0 to 1, skip 2
    
    # get transformed frame, only first returned value is the transformed frame
    frame_t = transform(frame)[0]
    
    # Convert this numpy array to torch tensor
    # Convert color space from Red(0), Blue(1), Green(2) to Green Red Blue
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    
    
    # Neural network takes batch inputs
    # First dimension corresponds to the batch
    # other dimension corresponds to the inputs
    # unsqeeze(index of the dimension of the batch)
    # unsqeeze(0) = first dimension
    x = Variable(x.unsqueeze(0))
    
    # feed transformed variable to neural networkF
    # get output y
    y = net(x)
    
    
    
    # Get specific detection, create a new tensor
    # y.data = values of the output
    detections = y.data
    
    
    # We need to normalize the tensor to 0 and 1
    # for this we scale our tensor into 4 dimensions
    scale = torch.Tensor([width, height, width, height])
    
    
    # detection
    for i in range(detections.size(1)): # For every class:
    
        # We initialize the loop variable j that will correspond to the 
        # occurrences of the class.
        j = 0 
        
        # We take into account all the occurrences j of the class i that 
        # have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6: 
            
            # We get the coordinates of the points at the upper left and the 
            # lower right of the detector rectangle.
            pt = (detections[0, i, j, 1:] * scale).numpy() 
            
            # We draw a rectangle around the detected object.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 
            
            # We put the label of the class right above the rectangle.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            
            # We increment j to get to the next occurrence.
            j += 1 
            
    # We return the original frame with the detector rectangle and the label 
    # around the detected object.        
    return frame 

# Creating the SSD neural network
# We create an object that is our neural network ssd.
net = build_ssd('test') 

# We get the weights of the neural network from another one that is pretrained 
# (ssd300_mAP_77.43_v2.pth).
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 

# Creating the transformation
# We create an object of the BaseTransform class, a class that will do the 
# required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 

# Doing some Object Detection on a video
# We open the video.
reader = imageio.get_reader('3s-Dhaka.mp4') 


# We get the fps frequence (frames per second).
fps = reader.get_meta_data()['fps'] 

# We create an output video with this same fps frequence.
writer = imageio.get_writer('output.mp4', fps = fps) 

# We iterate on the frames of the output video:
for i, frame in enumerate(reader): 
    
    # We call our detect function (defined above) to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 
    
    # We add the next frame in the output video.
    writer.append_data(frame) 
    
    # We print the number of the processed frame.
    print(i) 
    
# We close the process that handles the creation of the output video.    
writer.close() 
