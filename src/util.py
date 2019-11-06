#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utils file. Not all functions are used.
"""

import os,cv2,torch
import matplotlib.pyplot as plt
import itertools as itt
import numpy as np
import math



class Utils():
    
    def __init__(self):
        
        #Init for bbox calculations. Only looking at a 320 x 320 patch for speed.
        self.h1_list = [320,160,80,40,20,10,5,2.4] #Got from manual calculations
        self.w1_list = [320,160,80,40,20,10,5,2.4]
        
        #Levels
        self.var_per_level = 4 
        self.levels = 8
        
    
    def fetch_image(self,PATH,resize=True):
        """
        Function to fetch image for env.
        PATH needs to be a full path to an image.
        """
        new_dim = (1280,720)
        img = cv2.imread(PATH)
        self.ori_height = img.shape[0]
        self.ori_width = img.shape[1]
        
        if resize:
            img = cv2.resize(img,new_dim)
            
            #Store how much we've rescaled by
            self.resize_factor = [new_dim[1]/self.ori_height,
                                  new_dim[0]/self.ori_width]
            
        return img 
    
    
    def crop_image(self,img,tl_w,tl_h,br_w,br_h):
        """
        Called by take_action() in Agent's class to return a cropped image
        according to the action's bbox specification.
        """
        new = img[tl_h:br_h,tl_w:br_w,:]
        return new
        
    
    
    def crop_PIL(self,img,x,y,w,h):
        return img.crop((x,y,x+w,y+h))
        
    
    
    def writeim(self,img,seq_name,seq_idx,idx,scene,label):
        
        stl_w = int(scene[0])
        stl_h = int(scene[1])
        sbr_w = int(scene[2])
        sbr_h = int(scene[3])
        
        ltl_w = int(label[0]) + stl_w
        ltl_h = int(label[1]) + stl_h
        lbr_w = int(label[2]) + stl_w
        lbr_h = int(label[3]) + stl_h
        
        img_ = self.draw_bbox2_noshow(img,[stl_w,stl_h,sbr_w,sbr_h],[ltl_w,ltl_h,lbr_w,lbr_h])
        
        seq_dir = 'seq' + str(seq_idx)
        dirname = '/home/keifer/Documents/MachineLearning/BOBBY/v8_squeezenet_smlkernel/output/' + seq_dir + '/'
        img_name = 'outimg'+str(idx) +'.jpg'
        
        total_path = dirname + img_name        
        resp = cv2.imwrite(total_path,img_)
        
        return resp
    
    def imgshow(self,img):
        """
        Function to dispay an image.
        """
        
        #Include preprocess if any
        plt.imshow(img)
        plt.show()
    
    
    def fetch_truth(self,PATH,resize=True):
        """
        Function to fetch the bbox data for a sample set.
        PATH needs to be a full path to a particular seqeunce's csv file.
        
        This is different from fetching image because a file contains many bbox.
        """
        #!!! Remember to resize bbox since we've resized the picture.
        with open(PATH) as f:
            raw = f.readlines()
            
        #Parse and clean
        raw = [line.strip().split(',') for line in raw] #strip whitespace
        raw = [[int(num) for num in entry] for entry in raw] #convert to ints
        
        #If we resized
        if resize:
            for idx,entry in enumerate(raw):
                tl_w = round(entry[0] * self.resize_factor[1])
                tl_h = round(entry[1] * self.resize_factor[0])
                br_w = round(tl_w + (entry[2] * self.resize_factor[1]))
                br_h = round(tl_h + (entry[3] * self.resize_factor[0]))
                
                #Sub
                raw[idx] = [tl_w,tl_h,br_w,br_h]

        
        return raw
    
    
    def out2bbox(self,nn_out,ratio,scale):
        """
        Given final output from network, translate it to bbox coordinates for
        drawing, displaying and calculating loss.
        
        nn_out should be a list of (m,n) values with 4 values (2 sets of m,n)
        constitutng a single level. Total of 20 vars for loc, 5 for scale, 5 for ratio.
        
        Total level == 5
        
        Also note this is for image patches that are 
        assumed to be 320x320. Rescale them.
        
        Outputs a numpy array
        """

        #Parse through recursively.
        """
        Note that though the formulation has many different m and n, the process
        is an iterative one, and thus we could use a loop to do it instead as no
        intermediary values of m,n and coordinates would be needed.
        """
        #Init
        h0,w0 = 0,0
#        nn_out = nn_out.tolist() #Conv to list for index() method
        #Iteratively find top left (X,Y) and bottom right (X,Y)
        for idx,lvl in enumerate(range(self.levels)):
            
            #Extract out level's (m,n)
            state = nn_out[:4] #(m1,n1),(m2,n2) completely defines a single level
            m1,m2,n1,n2 = state
            nn_out = nn_out[4:] #Removed duplicates
            
            #Check if there are values or not, if not cont.
            #Will need to preprocess NN output since it may not be cleanly (0,1)
            if 1 not in state:
                continue
            
            #Fetch h1 and w1
            h1 = self.h1_list[idx]
            w1 = self.w1_list[idx]
            
            Hl = (h0*m1) + (h0 + (0.5*h1))*m2
            Hh = ((h0 + (0.5*h1))*m1) + ((h0 + h1)*m2)
            H_ = [Hl,Hh]
            
            Wl = (w0*n1) + (w0 + (0.5*w1))*n2
            Wh = ((w0 + (0.5*w1))*n1) + ((w0 + w1)*n2)
            W_ = [Wl,Wh]
            
            #Update h0,w0
            h0 = min(H_) #For h0,w0, we'll take the lowest number from previous level
            w0 = min(W_)            
            
            
        #Fetch and multply by ratio and scale. Leftover in nn_out are these 10 vars.
#        ratio = nn_out[32] #Continuous value since we're using reg here.
#        scale = nn_out[33]

    
        #Resize and rescale
        height = float(Hh - Hl)
        width = float(Wh - Wl)
        mid_h = float((height/2) + Hl)
        mid_w = float((width/2) + Wl)
        
        width_ratioed = width * ratio #Ratio apply to width only
        width_ = round(width_ratioed / scale) #It's div cause scale is < 1
        height_ = round(height / scale)
        
        #Move bbox to center of minigrid
        Hl = round(mid_h - (height_/2))
        Hh = round(mid_h + (height_/2))
        Wl = round(mid_w - (width_/2))
        Wh = round(mid_w + (width_/2))
        out = np.array([Wl,Hl,Wh,Hh])
        
        #Rectify negative and positive over-bound
        for idx,i in enumerate(out):
            if i < 0:
                out[idx] = 0
                
            elif i > 320:
                out[idx] = 320
        
        return out
        
    
    def draw_bbox(self,img,coord):
        """
        Draws and displays image with bbox.
        Image and coord should be a resized to 1280 x 720 version.
        """
        tl_w = int(coord[0])
        tl_h = int(coord[1])
        br_w = int(coord[2])
        br_h = int(coord[3])
        
        cv2.rectangle(img, (tl_w,tl_h), (br_w,br_h), (255,255,0), 2)
        cv2.imshow('Image',img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def draw_bbox2(self,img,coord1,coord2):
        """
        Draws and displays image with bbox.
        Image and coord should be a resized to 1280 x 720 version.
        """
        tl_w1 = int(coord1[0])
        tl_h1 = int(coord1[1])
        br_w1 = int(coord1[2])
        br_h1 = int(coord1[3])
        
        tl_w2 = int(coord2[0])
        tl_h2 = int(coord2[1])
        br_w2 = int(coord2[2])
        br_h2 = int(coord2[3])
        
        cv2.rectangle(img, (tl_w1,tl_h1), (br_w1,br_h1), (255,0,0), 2)
        cv2.rectangle(img, (tl_w2,tl_h2), (br_w2,br_h2), (255,255,0), 2)

        cv2.imshow('Image',img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return img
    
    
    def draw_bbox2_noshow(self,img,coord1,coord2):
        """
        Draws and displays image with bbox.
        Image and coord should be a resized to 1280 x 720 version.
        """
        tl_w1 = int(coord1[0])
        tl_h1 = int(coord1[1])
        br_w1 = int(coord1[2])
        br_h1 = int(coord1[3])
        
        tl_w2 = int(coord2[0])
        tl_h2 = int(coord2[1])
        br_w2 = int(coord2[2])
        br_h2 = int(coord2[3])
        
        cv2.rectangle(img, (tl_w1,tl_h1), (br_w1,br_h1), (255,0,0), 2)
        cv2.rectangle(img, (tl_w2,tl_h2), (br_w2,br_h2), (255,255,0), 2)
        
        return img
        

        
    def generate_bboxtemp(self):
        """
        Used in training in order to find the optimal bbox combination  
        (m1,n1,m2,n1,...) that matches the ground truth coordinates for loss
        calculation.
        
        The method is a greedy based method; generate all possible bbox's coord
        and use a LSE to find the best suiting bbox combination amongst the
        possibilities.
        
        No scale and ratio gen needed.
        """
        
        #For 5 levels, the possible bbox combination would be the following.        
        coord_comb = np.array([[1,0,1,0],
                                 [1,0,0,1],
                                 [0,1,1,0],
                                 [0,1,0,1]]) #All null (0,0,0,0) is ignored at this step.

        
        #Generate combination
        comb_len = 0
        combinations=[]

        
        for lvl in range(self.levels + 1)[1:]: #Skip lvl == 0    
            total_comb = []
            
            #Generate the required (m,n) sets wrt level.
            for i in range(lvl):    
                total_comb.append(coord_comb)
            
            
            #Generate coord comb
            coords = list(itt.product(*total_comb)) #Generate combinations
            
            #Pad coord
            for idx,cd in enumerate(coords):
                
                #Join elements in coords first
                cd = np.asarray(cd).flatten()
#                cd = cd[0]
                    
                #Get num of 0 to place
                full_len = self.levels * self.var_per_level
                cd_len = len(cd)
                pad = np.zeros((full_len-cd_len))
                
                coords[idx] = np.concatenate((cd,pad))
            
            
#            ratioed = list(itt.product(coords,ratio_comb))
#            ratioed = [np.concatenate((i[0],i[1])) for i in ratioed]
#            scaled = list(itt.product(ratioed,scale_comb))
#            combination = [np.concatenate((i[0],i[1])) for i in scaled]
            combinations.extend(coords)
            
        print("Generated {} unique bounding boxes label.".format(len(combinations)))
#            
    
        return combinations
                
                
    def iou(self,boxA,boxB):
        """
        Calculates the IoU of 2 bbox.
        Format of boxA and boxB should be [tl_x,tl_y,br_x,br_y].
        
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
         
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou


    def precision(self,pred,label,thres=30):
        """
        Calculate the precision of the bbox.
        Defined as the distance of the centroid of pred to centroid of label.
        """
        
        pred_cw = (pred[2] - pred[0])/2 + pred[0]
        pred_ch = (pred[3] - pred[1])/2 + pred[1]

        label_cw = (label[2] - label[0])/2 + label[0]
        label_ch = (label[3] - label[1])/2 + label[1]
        
        w_dis = (pred_cw - label_cw)**2
        h_dis = (pred_ch - label_ch)**2

        eucl_dis = math.sqrt(w_dis + h_dis)

        if eucl_dis >= thres:
            return 0
        
        else:
            if eucl_dis >= 0 and eucl_dis < 6:
                return 1
            
            elif eucl_dis >=6 and eucl_dis < 10:
                return 0.9
            
            elif eucl_dis >=10 and eucl_dis < 15:
                return 0.8
            
            elif eucl_dis >=15 and eucl_dis < 20:
                return 0.7
            
            elif eucl_dis >=20 and eucl_dis < 25:
                return 0.6
            
            elif eucl_dis >=25 and eucl_dis < 30:
                return 0.5
            
            elif eucl_dis >=30 and eucl_dis < 35:
                return 0.4
        
            elif eucl_dis >=35 and eucl_dis < 40:
                return 0.3
            
            elif eucl_dis >=40 and eucl_dis < 45:
                return 0.2
            
            elif eucl_dis >=45 and eucl_dis < 50:
                return 0.1
            
            else:
                return 0

    def unique_comb(self,ls):
        """
        Used to check for unique np array in a list.
        """
        str_ls = []
        sep = ','
        
        #Conv to list
        for i in ls:
            temp = i.tolist()
            temp_ = [str(j) for j in temp]
            _ = sep.join(temp_)
            str_ls.append(_)
            
        unique_element = set(str_ls)
        
        #Check if it's unique
        if len(unique_element) == len(ls):
            print("The combinations is unique")
            return None
        else:
            print("There are duplicate combinations!")
            return None
        
        return str_ls
            