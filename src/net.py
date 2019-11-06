#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:42:07 2019
@author: keifer
"""

import time,os,copy
import torch
from torchvision import models


class BOBBY2(nn.Module):
    

    def __init__(self,batch_size,action_size):
        super(BOBBY2,self).__init__()
        
        #NOTICE
        ##################### Init custom model ##################### 
        self.batch_size = batch_size
        self.action_size = action_size
        
        #Setup new net
        alex = models.alexnet(pretrained=True)
        self.FE_scene = nn.Sequential(*list(alex.children())[:1])


        self.Mixer = nn.Sequential(nn.Dropout2d(0.5),
                                   nn.Conv2d(1280,256,1,1),
                                   nn.ReLU(True))

        #Classifier  
        self.Classifier = nn.Sequential(nn.Linear(9216,2000),
                                        nn.Dropout(0.5),
                                        nn.ReLU(True),
                                        nn.Linear(2000,500),
                                        nn.Dropout(0.5),
                                        nn.ReLU(True),
                                        nn.Linear(500,self.action_size))




    def freeze_FE_scene(self):
        """
        Simple fx called to freeze the FE part of net.
        """
        temp = []

        for param in self.FE_scene.parameters():
            param.requires_grad = False 

        #Verify
        for param in self.FE_scene.parameters():
            _ = param.requires_grad == False
            temp.append(_)

        assert False not in temp, "Error! Not all of FE_scene layers are frozen!"

        print("FE_scene is now frozen.")



    def freeze_Mix(self):
        """
        Simple fx called to unfreeze the Mixer part of net.
        """
        temp = []

        for param in self.Mixer.parameters():
            param.requires_grad = False 

        #Verify
        for param in self.Mixer.parameters():
            _ = param.requires_grad == False
            temp.append(_)

        assert False not in temp, "Error! Not all of Mixer layers are frozen!"

        print("Mixer is now frozen.")



    def unfreeze_FE_scene(self,depths):
        """
        Simple fx called to freeze the FE part of net.
        Depths is a list containing the param indices to unfreeze.
        """
        for idx,param in enumerate(self.FE_scene.parameters()):

            if idx in depths:
                param.requires_grad = True

        #Verify
        for idx, param in enumerate(self.FE_scene.parameters()):

            if param.requires_grad == True:
                print("Layer {} of FE_scene is thawed.".format(idx))
            

    def unfreeze_Mix(self):
        """
        Simple fx called to unfreeze the Mixer part of net.
        """
        temp = []

        for param in self.Mixer.parameters():
            param.requires_grad = True 

        #Verify
        for param in self.Mixer.parameters():
            _ = param.requires_grad == True
            temp.append(_)

        assert False not in temp, "Error! Not all of Mixer layers are thawed!"

        print("Mixer is now thawed.")


    def unfreeze_cls(self,depths):
        """
        Simple fx called to unfreeze the Classifier part of net.
        Depths is a list containing the param indices to unfreeze.
        """
        for idx,param in enumerate(self.Classifier.parameters()):

            if idx in depths:
                param.requires_grad = True

        #Verify
        for idx, param in enumerate(self.Classifier.parameters()):

            if param.requires_grad == True:
                print("Layer {} of Classifier is thawed.".format(idx))


    def freeze_cls(self):
        """
        Simple fx called to freeze the Classifier part of net.
        """
        temp = []

        for param in self.Classifier.parameters():
            param.requires_grad = False 

        #Verify
        for param in self.Classifier.parameters():
            _ = param.requires_grad == False
            temp.append(_)

        assert False not in temp, "Error! Not all of Classifier layers are frozen!"

        print("Classifier is now frozen.")

    

    def forward(self,x_scene,x_exem,exem=False):
        """
        Note batch dim omitted for simplicity.

        Inputs:
            x_scene: Input scene with or without target. Size of [3x224x224]
            x_exem: Input of exemplars concatenated channel-wise.
            exem: TRUE to perform featurization of raw exemplars, FALSE for normal tracking with featurized exemplars.

        Return:
            out (exem = FALSE): [1x6] vector comprised of [1x4] unnormalized bounding box coordinates and [1x2] objectness score.
            out (exem = TRUE): Featurized exemplar. Size of [(256*buffer_size)x6x6]
        """
        
        if not exem:

            #Feature extract
            FE1_out = self.FE_scene(x_scene)
            
            #Concat and mix
            FE_cat = torch.cat((FE1_out,x_exem),dim=1) #Concat along depth
            FE_mixed = self.Mixer(FE_cat)
            mix_resized = FE_mixed.view(self.batch_size,-1)
 
            #Classify
            out = self.Classifier(mix_resized)
        
        else:

            #A buffer cache iteration. Featurize exemplar.
            out = self.FE_scene(x_exem)
        
        return out
    


    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
            
    
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])



