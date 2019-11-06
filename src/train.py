#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time,os,copy
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import data_utils as du_ilsvrc_nsq
from lr_finder import *
from pathlib import Path
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from util import Utils
from net import BOBBY2
from torch.optim.lr_scheduler import _LRScheduler
from onecycle import OneCycle as OC


    
    
def gen_dl(dataset,batch_size,loader_workers):
    """
    Quick helper function to return dataloader.
    """
    return DataLoader(dataset, batch_size=batch_size,
                            drop_last = True, shuffle=True, 
                            num_workers=loader_workers)



class oc_sched(_LRScheduler):
    """
    Quick onecycler LR scheduler riding on top of OC from Nachiket Tanksale 
    https://github.com/nachiket273/One_Cycle_Policy
    """

    def __init__(self, optimizer, max_lr, num_iter, last_epoch=-1):
        self._lrs = OC(num_iter,max_lr)
        super(oc_sched, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        lr,mom = self._lrs.calc()

        # # print(self.base_lrs)
        # print(lr,mom)

        return [lr for base_lr in self.base_lrs]



def _featurize(inputs,model):
    """
    Helper function used to featurize exemplars before feeding into
    buffer.
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(*inputs).detach() #Featurize raw exem

    return outputs



def train(model,optimizer,bbox_crit,obj_crit,sched,train_dl,val_dl,bs=160, alpha=1,beta=1):
    """
    Training loop function.

    Input:
        model: BOBBY2.
        optimizer: optimizer used.
        bbox_crit: Bounding box criterion.
        obj_crit: Objectness criterion.
        scehd: Training lr scheduler.
        train_dl: Train dataloader.
        val_dl: Validation dataloader.
        bs: Batch size; default 160.
        alpha: Objectness loss multiplier; default 10.
        beta: Ranges from [0,1]. Used to control portion of dataset to use.

    Return:
        model: Trained model.
        avg_train_loss: Average training loss.
        lrs: History of learning rate. Used for debugging.
        epoch_tloss_ls: List of batch total losses. Sorry for the bad name :D. Update later.
        epoch_bboxloss_ls: List of batch bbox losses. Sorry for the bad name :D. Update later.
        epoch_objloss_ls: List of batch objectness losses. Sorry for the bad name :D. Update later.
        valid_losses: List of total valid losses. Sorry for the bad name :D. Update later.
        valid_bbloss: List of valid bbox losses. Sorry for the bad name :D. Update later.
        valid_objloss: List of valid objectness losses. Sorry for the bad name :D. Update later.
        sched: Training lr scheduler.
        optimizer: optimizer used.
    """

    #Init
    device = 'cuda'
    
    #Logging init   
    epoch_tloss_ls = []
    epoch_bboxloss_ls, epoch_objloss_ls = [],[]
    valid_losses = []
    valid_bbloss, valid_objloss = [],[]
    lrs = []
    
    #Losses counter
    val_int = 50 #Take X batches only of validation data for validation
    train_loss = 0 
    total_obj_loss = 0
    total_bbox_loss = 0
    time_start = time.time()
    model.train()
    
    #Shorter train cycle
    len_dl = len(train_dl)
    ijx_break = round(len_dl*beta)
    
    #Train
    with torch.set_grad_enabled(True):
        for ijx,smp in enumerate(train_dl):

            #zero the parameter gradients
            optimizer.zero_grad()
            
            #Get a new set of inputs and labels
            scene = smp['Img']['scene'] #Returns a scene tensor
            exem = smp['Img']['exem'] #Returns list of exems tensors           
            bbox = smp['Annot']['bbox']
            obj = smp['Annot']['obj']
            
            #Featurize exem first
            with torch.no_grad():
                exem_bt = []
                for i in range(len(exem)):
                    payload = (0,exem[i].to(device),True)
                    exem_f = _featurize(payload,model)
                    exem_f = torch.cat((exem_f[0],exem_f[1],exem_f[2],exem_f[3]))
                    exem_bt.append(exem_f)
                exem_bt = torch.stack(exem_bt)

            #Parse label and pack input to model
            bbox = du_vis.parse_ant(bbox).cuda() #parse_ant in du_vis and du_ilsvrc are the same
            obj = du_vis.parse_ant(obj).cuda()
            payload = (scene.to(device),exem_bt.to(device),False)

            #Printing log to check LR during training
            for pidx,param_group in enumerate(optimizer.param_groups):
                lr_param = param_group['lr']
                lrs.append(lr_param)


            #Forward
            outputs = model(*payload)
            out_bbox = outputs[:,:4] * obj.float() #Float to multiply with outputs
            out_obj = outputs[:,4:]
            obj = obj.long() #Back to long as needed by CELoss
            bbox = bbox * obj.float() #Mul truth bbox as well to match out_bbox.
        
            #Compute bbox
            bbox_loss = bbox_crit(out_bbox,bbox)
            obj_loss = obj_crit(out_obj,obj.squeeze(1)) * alpha
            total_loss = bbox_loss + obj_loss
   
            # Backward pass
            total_loss.backward()
            optimizer.step()
            sched.step()
            
            #Loss stats logging
            train_loss += total_loss.item() * obj.size(0)  
            epoch_tloss_ls.append(total_loss.item())
            epoch_bboxloss_ls.append(bbox_loss.item())
            epoch_objloss_ls.append(obj_loss.item())


            ##Validation
            if ijx % val_int == 0:
                
                #Validate
                model,val_loss,vbb_loss,vobj_loss = validate(model,bs,val_dl,bbox_crit,obj_crit,alpha)

                valid_losses.append(val_loss)
                valid_bbloss.append(vbb_loss)
                valid_objloss.append(vobj_loss)

                time_intermit = time.time() - time_start

                print("Currently at Iter {}/{} | Time Elapsed: {}s".format(ijx+1,len(train_dl),time_intermit))
                print("Current Avg Train Total-Loss: {}".format(total_loss.item()))
                print("Current Avg Train Bbox-Loss: {}".format(bbox_loss.item()))
                print("current AVg Train Obj-Loss: {}".format(obj_loss.item()))
                
                print("Current Val Total-Loss: {}".format(val_loss))
                print("Current Val Bbox-Loss: {}".format(vbb_loss))
                print("Current Val Obj-Loss: {}".format(vobj_loss))
                
            
            #Stop early
            if ijx == ijx_break:
                break
                            
            
    #End of training stats compute and printout
    avg_train_loss = train_loss / ((ijx+1) * bs)
    time_end = time.time() - time_start

    print("*"*50)
    print("Epoch Train Total-Loss: {}".format(train_loss))
    print("Avg Train Total-Loss: {}".format(avg_train_loss))
    print("Last Total-Loss: {}".format(total_loss))
    print("Last Bbox-Loss: {}".format(bbox_loss))
    print("Last Obj-Loss: {}".format(obj_loss))
    
    print("Last Val Total-Loss: {}".format(val_loss))
    print("Last Val Bbox-Loss: {}".format(vbb_loss))
    print("Last Val Obj-Loss: {}".format(vobj_loss))
    print("Total Training Time: {}sec / {}min / {}hr".format(time_end,(time_end/60),(time_end/60/60)))
    print("*"*50)

    
    return (model, avg_train_loss, lrs, epoch_tloss_ls, epoch_bboxloss_ls,epoch_objloss_ls,
            valid_losses, valid_bbloss,valid_objloss, sched, optimizer)


def validate(model,bs,val_dl,bbox_crit,obj_crit,stop=5,alpha=1):
    """
    Validation function. I/O similar to train. Please refer to the training
    function for more info.
    """

    device = 'cuda'
    val_loss = 0

    model.eval()
    epoch_tloss,epoch_bbloss, epoch_objloss = [],[],[]
    
    with torch.set_grad_enabled(False):
        for ijx,smp in enumerate(val_dl):

            #Get a new set of inputs and labels
            scene = smp['Img']['scene'] #Returns a scene tensor
            exem = smp['Img']['exem'] #Returns list of exems tensors           
            bbox = smp['Annot']['bbox']
            obj = smp['Annot']['obj']

            #Featurize exem first
            with torch.no_grad():
                exem_bt = []
                for i in range(len(exem)):
                    payload = (0,exem[i].to(device),True)
                    exem_f = _featurize(payload,model)
                    exem_f = torch.cat((exem_f[0],exem_f[1],exem_f[2],exem_f[3]))
                    exem_bt.append(exem_f)
                exem_bt = torch.stack(exem_bt)

            #Parse label and pack input to model
            bbox = du_vis.parse_ant(bbox).cuda() #parse_ant in du_vis and du_ilsvrc are the same
            obj = du_vis.parse_ant(obj).cuda()
            payload = (scene.to(device),exem_bt.to(device),False)
            

            #Forward
            outputs = model(*payload)
            out_bbox = outputs[:,:4] * obj.float() #Float to multiply with outputs
            out_obj = outputs[:,4:]
            obj = obj.long() #Back to long as needed by CELoss
            bbox = bbox * obj.float() #Mul truth bbox as well to match out_bbox.
        
            #Compute bbox
            bbox_loss = bbox_crit(out_bbox,bbox)
            obj_loss = obj_crit(out_obj,obj.squeeze(1)) * alpha
            total_loss = bbox_loss + obj_loss            
            
            #Log loss stats
            epoch_tloss.append(total_loss.item())
            epoch_bbloss.append(bbox_loss.item())
            epoch_objloss.append(obj_loss.item())
            
            # statistics
            val_loss += total_loss.item() * obj.size(0)
            
            #Let's take 3 batches only for each val
            if ijx > stop:
                break
            

    avg_val_loss = val_loss / ((ijx+1) * bs)
    avg_obj_loss = np.mean(epoch_objloss)
    avg_bbox_loss = np.mean(epoch_bbloss)
    
    print("*"*30)
    print("Epoch Val Total-Loss: {}".format(val_loss))
    print("Avg Val Total-Loss: {}".format(avg_val_loss))
    print("Avg Val Bbox Loss: {}".format(avg_bbox_loss))
    print("Avg Val Obj Loss: {}".format(avg_obj_loss))
    print("*"*30)
    
    
    return model, avg_val_loss, avg_bbox_loss, avg_obj_loss



def evaluate(model,bs,val_dl,bbox_crit,obj_crit,stop=5,alpha=1):
    """
    Function used to run evaluation of model on dataset.
    Calculates and returns IoU and Precision scores.

    I/O similar to train. Please refer to the training function for more info.
    """

    device = 'cuda'
    val_loss = 0

    model.eval()
    epoch_tloss,epoch_bbloss, epoch_objloss = [],[],[]
    iou_ls, prec0_ls, prec10_ls, prec20_ls, prec30_ls, prec40_ls, prec50_ls, prec60_ls = [],[],[],[],[],[],[],[]
    tgt_ls = []
    
    with torch.set_grad_enabled(False):
        for ijx,smp in enumerate(val_dl):

            #Get a new set of inputs and labels
            scene = smp['Img']['scene'] #Returns a scene tensor
            exem = smp['Img']['exem'] #Returns list of exems tensors           
            bbox = smp['Annot']['bbox']
            obj = smp['Annot']['obj']

            #Featurize exem first
            with torch.no_grad():
                exem_bt = []
                for i in range(len(exem)):
                    
                    
                    print("EXEM: {}".format(exem[i].shape))
                    
                    payload = (0,exem[i].to(device),True)
                    exem_f = _featurize(payload,model)
                
                    print("MID EXEM_F: {}".format(exem_f.shape))
                
                    exem_f = torch.cat((exem_f[0],exem_f[1],exem_f[2],exem_f[3]))
                    exem_bt.append(exem_f)
                exem_bt = torch.stack(exem_bt)
                
                
                print(exem_f.shape,exem_bt.shape)

            #Parse label and pack input to model
            bbox = du_vis.parse_ant(bbox).cuda() #parse_ant in du_vis and du_ilsvrc are the same
            obj = du_vis.parse_ant(obj).cuda()
            payload = (scene.to(device),exem_bt.to(device),False)
            
            #Forward
            outputs = model(*payload)
            out_bbox = outputs[:,:4] * obj.float() #Float to multiply with outputs
            out_obj = outputs[:,4:]
            obj = obj.long() #Back to long as needed by CELoss
            bbox = bbox * obj.float() #Mul truth bbox as well to match out_bbox.
        
            #Compute bbox
            bbox_loss = bbox_crit(out_bbox,bbox)
            obj_loss = obj_crit(out_obj,obj.squeeze(1)) * alpha
            total_loss = bbox_loss + obj_loss            
            
            #Log loss stats
            epoch_tloss.append(total_loss.item())
            epoch_bbloss.append(bbox_loss.item())
            epoch_objloss.append(obj_loss.item())
            
            #Statistics
            val_loss += total_loss.item() * obj.size(0)
            
            #IoU and Precision            
            iou = [util.iou(p,l) for p,l in zip(out_bbox,bbox)]
            prec_0 = [util.precision(p,l,0) for p,l in zip(out_bbox,bbox)]
            prec_10 = [util.precision(p,l,10) for p,l in zip(out_bbox,bbox)]
            prec_20 = [util.precision(p,l,20) for p,l in zip(out_bbox,bbox)]
            prec_30 = [util.precision(p,l,30) for p,l in zip(out_bbox,bbox)]
            prec_40 = [util.precision(p,l,40) for p,l in zip(out_bbox,bbox)]
            prec_50 = [util.precision(p,l,50) for p,l in zip(out_bbox,bbox)]
            prec_60 = [util.precision(p,l,60) for p,l in zip(out_bbox,bbox)]
            tgt = [True if [*aaa].index(max(aaa)) == int(bbb) else False for aaa,bbb in zip(out_obj.cpu().numpy(),obj)]
            
            iou_ls.extend(iou)
            prec0_ls.extend(prec_0)
            prec10_ls.extend(prec_10)
            prec20_ls.extend(prec_20)
            prec30_ls.extend(prec_30)
            prec40_ls.extend(prec_40)
            prec50_ls.extend(prec_50)
            prec60_ls.extend(prec_60)
            tgt_ls.extend(tgt)
            
            
            #Let's take 3 batches only for each val
            if ijx > stop:
                break
            

    avg_val_loss = val_loss / ((ijx+1) * bs)
    avg_obj_loss = np.mean(epoch_objloss)
    avg_bbox_loss = np.mean(epoch_bbloss)
    
    print("*"*30)
    print("Epoch Val Total-Loss: {}".format(val_loss))
    print("Avg Val Total-Loss: {}".format(avg_val_loss))
    print("Avg Val Bbox Loss: {}".format(avg_bbox_loss))
    print("Avg Val Obj Loss: {}".format(avg_obj_loss))
    print("*"*30)
    
    
    return {'avg total loss':avg_val_loss, 'avg bbox loss':avg_bbox_loss,
            'avg obj loss': avg_obj_loss, 'iou':iou_ls, 'p0':prec0_ls, 'p10':prec10_ls,
            'p20':prec20_ls, 'p30':prec30_ls, 'p40':prec40_ls, 'p50':prec50_ls,'p60':prec60_ls,
            'objectness':tgt_ls,'tloss_ls':epoch_tloss,'bbloss_ls':epoch_bbloss,'objloss_ls':epoch_objloss}



#Train
if __name__ == "__main__":

    #Init libs
    util = Utils()  
    
    
    #Net hyperparams
    batch_size = 32
    loader_workers = 4 
    ex_int = 60
    percent_neg = 0.5
    action_size = 6 #4 for bbox 2 for cls (CrossEntropyLoss)
    net = BOBBY2(batch_size,action_size).cuda()
    
    #Save dir
    save_dir = None
    
    #Define file path for Ilsvrc (Train)
    ilv_train_img_root = Path('/mnt/02D2CC5FD2CC5895/Machine Learning/dataset/ilsvr_vid/ILSVRC2017_VID/ILSVRC/Data/VID/')
    ilv_train_ant_root = Path('/mnt/02D2CC5FD2CC5895/Machine Learning/dataset/ilsvr_vid/ILSVRC2017_VID/ILSVRC/Annotations/VID/')
    ilv_train_posneg_ls = 'ilsvrc_train_csv/posneg.csv'
    ilv_train_pos_ls = 'ilsvrc_train_csv/ilsvrc_vanilla_pos.csv'
    ilv_train_neg_ls = 'ilsvrc_train_csv/ilsvrc_vanilla_neg.csv'
    ilv_train_cusneg_ls = 'ilsvrc_train_csv/neg_below08.csv'
    
    
    #Define file path for Ilsvrc (Val)
    ilv_val_img_root = Path('/mnt/02D2CC5FD2CC5895/Machine Learning/dataset/ilsvr_vid/ILSVRC2017_VID/ILSVRC/Data/VID/')
    ilv_val_ant_root = Path('/mnt/02D2CC5FD2CC5895/Machine Learning/dataset/ilsvr_vid/ILSVRC2017_VID/ILSVRC/Annotations/VID/')
    ilv_val_posneg_ls = 'ilsvrc_val_csv/posneg.csv'
    ilv_val_pos_ls = 'ilsvrc_val_csv/ilsvrc_vanilla_pos.csv'
    ilv_val_neg_ls = 'ilsvrc_val_csv/ilsvrc_vanilla_neg.csv'
    ilv_val_cusneg_ls = 'ilsvrc_val_csv/neg_below08.csv'
    
    
    #Initialize dataset
    ilv_train_dt = du_ilsvrc_nsq.gen_dt(ilv_train_img_root, ilv_train_ant_root, ilv_train_posneg_ls,
                                ilv_train_pos_ls, ilv_train_neg_ls, ilv_train_cusneg_ls, 
                                percent_neg=percent_neg, ex_int=ex_int, transform=True)
    ilv_val_dt = du_ilsvrc_nsq.gen_dt(ilv_val_img_root, ilv_val_ant_root, ilv_val_posneg_ls,
                                ilv_val_pos_ls, ilv_val_neg_ls, ilv_val_cusneg_ls, 
                                percent_neg=percent_neg, ex_int=ex_int, transform=False)
    
    
    # Initialize dataloaders
    ilv_train_dl = gen_dl(ilv_train_dt,batch_size,loader_workers)
    ilv_val_dl = gen_dl(ilv_val_dt,batch_size,loader_workers)
    
    
    #Training settings and LR Cycle
    alpha = 10
    beta = 1
    epochs = 1
    train_len = len(ilv_train_dt)
    cycles = 1
    lr_cycle = int(((train_len//batch_size) - 1) * epochs * beta / cycles)
    

    #Init Losses. Note that these losses already avg over batch size by default.
    bbox_crit = nn.SmoothL1Loss(reduction='mean')
    obj_crit = nn.CrossEntropyLoss(reduction='mean')
    criterion = [bbox_crit,obj_crit] #For lr_learner
    optimizer = optim.Adam(net.parameters(), lr=1e-6,weight_decay=1e-2)

    
    #Pre-Train Printouts
    print("------------------------- Runtime METAS -------------------------")
    print("Run with Buffer interval of {}.".format(ex_int))
    print("Run with {}% of negative exems in Buffer.".format(percent_neg*100))
    print("-----------------------------------------------------------------")
    
    #Load net.
    net.load_state_dict(torch.load("/mnt/02D2CC5FD2CC5895/Machine Learning/BOBBY/v13_bigdata/checkpoints_7/0_smaller.pth"))
    print("Net Loaded.")

    #Freeze and unfreeze.
    net.freeze_FE_exem()
    net.freeze_FE_scene()

    #Find good LR. Uncomment to use.
    # lr_finder = LRFinder(net, optimizer, criterion, device="cuda", alpha = alpha)
    # lr_finder.range_test(ilv_val_dl, end_lr=10, num_iter=300)
    # lr_finder.plot()
    
    #Setup scheduler.
    learning_rate = 3e-3 #Use lr_finder above to determine.
    sched = oc_sched(optimizer,learning_rate,lr_cycle)

    #Train Log placeholders
    lrs_log = []
    t_loss_ls,v_loss_ls = [],[]
    t_bbox_loss_ls,t_obj_loss_ls = [],[]
    v_bbox_loss_ls,v_obj_loss_ls = [],[]
    worst_loss = 1000e1000

    
    #Train Loop
    for i in range(epochs):   
                
        #Train
        train_return = train(net,optimizer,bbox_crit,obj_crit,sched,ilv_train_dl,ilv_val_dl,bs=batch_size, alpha=alpha, beta=beta)
        net,avg_train_loss,lrs,ep_tloss,ep_bbloss,ep_objloss,ep_vloss, ep_vbbloss, ep_vobjloss, sched, optimizer = train_return
          
        #Log
        lrs_log.extend(lrs)
        t_loss_ls.extend(ep_tloss)
        v_loss_ls.extend(ep_vloss)
        t_bbox_loss_ls.extend(ep_bbloss)
        t_obj_loss_ls.extend(ep_objloss)
        v_bbox_loss_ls.extend(ep_vbbloss)
        v_obj_loss_ls.extend(ep_vobjloss)
    
        plt.plot(t_loss_ls)
        plt.plot(v_loss_ls)
        plt.show()
        
        #Check to save
        if v_loss_ls[-1] < worst_loss or i % 1 == 0:
            print("[---------------- Found Better Model! ----------------]")
            print("Previous Loss: {} | Current Loss: {}".format(worst_loss,v_loss_ls[-1]))
            best_model_wts = copy.deepcopy(net.state_dict())
            worst_loss = v_loss_ls[-1]
            print("[-----------------------------------------------------]")
            torch.save(net.state_dict(),save_dir)


    print("End of training.")