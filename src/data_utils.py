import torch, torchvision
import os, PIL, random, csv
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path



def compile_imgs(root_dir):
    """
    Deprecated. Used previously when custom Dataset compiles sample paths on instatiation.
    Current custom dataset instead accepts a pre-cooked path list to pos/neg samples.
    
    Use compile_data.
    """

    _ = [root_dir/i for i in os.listdir(root_dir)]
    heap_main = [root_dir/j/i for j in _ for i in os.listdir(j)] #These are folders for 3862 train seqs
    heap_main.sort()
    heap = [i/j for i in heap_main for j in os.listdir(i)]
    heap.sort()
        
    return heap


def compile_annots(root_dir):
    """
    Deprecated. Used previously when custom Dataset compiles sample paths on instatiation.
    Current custom dataset instead accepts a pre-cooked path list to pos/neg samples.
    
    Use compile_data.
    """
    
    _ = [root_dir/i for i in os.listdir(root_dir)]
    heap_main = [root_dir/j/i for j in _ for i in os.listdir(j)] #These are folders for 3862 train seqs
    heap_main.sort()
    heap = [i/j for i in heap_main for j in os.listdir(i)]
    heap.sort()
        
    return heap


def compile_data(img_root,ant_root,posneg_ls,pos_ls,neg_ls,neg_ls1,seed=5):
    """
    Function that returns a dataset (hardcoded list) of pos and neg samples.
    Returns 2 lists: img_ls and annot_ls.
    
    Pos Sample: Translate and map idx from posneg.csv to 
    """
    
    ant_heap = []
    img_heap = []
    
    #Read csv
    posneg = parse_csv(posneg_ls)
    vanilla_pos = parse_csv(pos_ls)
    vanilla_neg = parse_csv(neg_ls)
    gen_neg = parse_csv(neg_ls1)
    
    #Random shuffle custom to be generated negative samples for representation.
    random.seed(seed)
    random.shuffle(gen_neg)
    
    #Idx for counting 
    vp,vn,gn = 0,0,0
    
    #Parse main list
    for i in posneg:
        
        #If it's neg
        if i == 0 and vn <= len(vanilla_neg)-1:
            _ = [0,Path(vanilla_neg[vn])]
            vn += 1
            
        #If it's neg exceeding vanilla neg
        if i == 0 and vn > len(vanilla_neg)-1:
            _ = [0,Path(gen_neg[gn])]
            gn += 1
            
        #If it's pos
        if i == 1:
            _ = [1,Path(vanilla_pos[vp])]
            vp += 1
            
        ant_heap.append(_)
        
        
        #Compute equal for imgs list
        ant_base = Path(_[1])
        ant_parts = ant_base.parts[-4:-1]
        name = ant_base.stem + '.JPEG'
        img = img_root/Path(*ant_parts)/Path(name)
        
        img_heap.append([i,img])
    
    
    return img_heap,ant_heap
    
    
def parse_csv(file):
    """
    Helper function that takes a csv path and return lists with csv content.
    """
    heap = []
    
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            
            try:
                heap.append(int(float(*row)))
            except:
                heap.append(*row)
        print(f'Processed {file} lines.')
        
    return heap
    


def check_trans(img):
    """
    Process and displays a batch of transformed scene and exems.
    """
    simg = img.permute(1,2,0).cpu()          

    #Plotting
    plt.imshow(simg)
    plt.show()



def rszasneed(full_img,annot):
    """
    Helper function that takes a fullscene and the annotations to resize the scene randomly for augmentation and
    returns the proper annotations.

    Also accounted for the cases where exem (defined by annot) is larger than the current scene patch dimension.
    """

    #Set patch size
    patch_sz = (360,360)
    
    #Size it
    img_w, img_h = full_img.size
    ex_tw, ex_th, ex_bw, ex_bh = annot[0], annot[1], annot[0] + annot[2], annot[1] + annot[3]
    
    #Setup resize range
    ratio_w = annot[2]/patch_sz[0]
    ratio_h = annot[3]/patch_sz[1]
    sz_max = max(ratio_w,ratio_h) #See which dim is the largest, that'll be the max resize up.
    
    if ratio_w <1 and ratio_h < 1: #If the exem is by default smaller than patch
        #Random resize that zooms and shrinks
        sz_fc = random.uniform(sz_max+0.5,1.5) #Make sure exem won't be larger than patch. +0.1 buffer
        new_w = img_w / sz_fc
        new_h = img_h / sz_fc

    elif ratio_w >= 1 or ratio_h >= 1: #If exem is larger than patch in any dim at all
        #Resize so sz of exem < sz of patch
        sz_fc = random.uniform(sz_max+0.1,sz_max+0.5) #Shrink more (max 3) since exem is large
        new_w = img_w / sz_fc
        new_h = img_h / sz_fc
        
    #Resize img and annot
    img = full_img.resize((round(new_w),round(new_h)),resample=PIL.Image.BICUBIC)
    ex_tw = ex_tw / sz_fc
    ex_th = ex_th / sz_fc
    ex_bw = ex_bw / sz_fc
    ex_bh = ex_bh / sz_fc
    annot = (ex_tw,ex_th,ex_bw,ex_bh)

    #Checks
    w = ex_bw - ex_tw
    h = ex_bh - ex_th
    
    assert w < patch_sz[0], "Error! The exem w is larger than patch_w | w: {}, patch_w: {}".format(w,patch_sz[0])
    assert h < patch_sz[1], "Error! The exem h is larger than patch_h | h: {}, patch_h: {}".format(h,patch_sz[1])

    return img, annot


def scene_crop_neg(full_scene,annot,scene_dim=360):
    """
    Helper function used in gen_dt to extracte a negative 360x360 patch from full scene.
    Uses the to_square_neg function since it'll work; for both vanilla and custom negatives.
    """
    #Crop square. Scene_dim dictates the shape of scene and the GAP on each size of a scene needed.
    scene,ant = to_square_neg(annot,full_scene,scene_dim)
    
    #Resize scene (360x360) crop to 224x224 as needed by net.
    scene = scene.resize((224,224),resample=PIL.Image.BICUBIC)
    
    #No need to compensate ant since negative smp has (0,0,0,0) ants.
    #     
    return scene,ant



def scene_crop(full_scene,annot,scene_dim=360):
    """
    Helper function used in gen_dt to extracte a positive 360x360 patch from full scene.
    """

    #Normalize dim and exem location in scene. Determine the gap on each side before crop.        
    full_scene,annot = rszasneed(full_scene,annot)
    img_w, img_h = full_scene.size
    ex_tw, ex_th, ex_bw, ex_bh = annot[0], annot[1], annot[2], annot[3] #Already added up in rszasneed
    nex_tw, nex_th, nex_bw, nex_bh = ex_tw/img_w, ex_th/img_h, ex_bw/img_w, ex_bh/img_h #normalized exem 
        
    ###Required scene patch
    req = (scene_dim/img_w, scene_dim/img_h)
    
    #Only do compute_cc padding if needed patch sz fits in the full scene
    if req[0] <= 1 and req[1] <= 1:
        tw_n,th_n,bw_n,bh_n = compute_cc(nex_tw,nex_th,nex_bw,nex_bh,req)
        
        #Compensate 
        tw = tw_n * img_w
        th = th_n * img_h
        bw = bw_n * img_w
        bh = bh_n * img_h

        #Crop. Needs to be PIL image.
        cropped = full_scene.crop((tw,th,bw,bh))
        rsz_fc1 = cropped.size[0]/224 #Need to return a 224 img anyhow
        rsz_fc2 = cropped.size[1]/224
        cropped = cropped.resize((224,224),resample=PIL.Image.BICUBIC) #Resize
        
        #Compensate annotations. Clip.
        ant_tw = annot[0] - tw
        ant_th = annot[1] - th
        ant_bw = annot[2] - tw
        ant_bh = annot[3] - th

        #Compensate annotations. Resize val.
        ant_tw,ant_bw = ant_tw/rsz_fc1, ant_bw/rsz_fc1
        ant_th,ant_bh = ant_th/rsz_fc2, ant_bh/rsz_fc2 
        ant_ = [i if i <= 224 else 224 for i in [ant_tw,ant_th,ant_bw,ant_bh]] 

    else:
        #Otherwise use backup pseudo-optimal strat of max square cut with min scretching.
        cropped, ant_ = to_square_scene(to_visfmt(annot),full_scene)
        
    return cropped, ant_



def compute_cc(nex_tw,nex_th,nex_bw,nex_bh,req):
    """
    Computes the spacing on each side of an exemplar for cropping. 
    Returns normalized coordinates to crop with.

    If overflows happens in two sides of a same dimension (e.g. scene size req is larger than entire full scene)
    the function will return the largest square image possible covering the exemplar. Make sure to have a resize
    catching such cases on the return of this function.
    """

    scene_w, scene_h = req[0], req[1]

    #Compute exem dim
    exem_w = nex_bw - nex_tw
    exem_h = nex_bh - nex_th
        
    #Catch problematic inputs
    assert scene_w > exem_w, "Error! The scene patch asked for is smaller than the exemplar. scene_w:{},exem_w:{}".format(scene_w,exem_w)
    assert scene_h > exem_h, "Error! The scene patch asked for is smaller than the exemplar. scene_h:{},exem_h:{}".format(scene_h,exem_h)
    assert req[0] <= 1, "Error! Patch size asked for is bigger than the actual pic. req[0]: {}".format(req[0])
    assert req[1] <= 1, "Error! Patch size asked for is bigger than the actual pic. req[1]: {}".format(req[1])

    #Size the gap needed
    req_w = scene_w - exem_w
    req_h = scene_h - exem_h

    #Randomize translation
    spf1 = random.uniform(0,1) #Split factor
    req_w1 = req_w * spf1
    req_w2 = req_w - req_w1

    spf2 = random.uniform(0,1)
    req_h1 = req_h * spf2
    req_h2 = req_h - req_h1

    #Check which side overflows
    ov_left = True if nex_tw < req_w1 else False 
    ov_right = True if (nex_bw + req_w2) > 1 else False
    ov_top = True if nex_th < req_h1 else False
    ov_bottom = True if (nex_bh + req_h2) > 1 else False

    ov_FLAGS = [ov_left,ov_top,ov_right,ov_bottom]
    ov = [req_w1-nex_tw, req_h1-nex_th, (nex_bw + req_w2)-1,(nex_bh + req_h2)-1] #How much spill over

    need_comp = True if any(ov_FLAGS) else False
    
    #Default cropping with no spillage
    new_th = nex_th - (req_h1)
    new_bh = nex_bh + (req_h2)
    new_tw = nex_tw - (req_w1)
    new_bw = nex_bw + (req_w2)
    output = [new_tw,new_th,new_bw,new_bh]

    #Comp needed
    if need_comp:
        ncomp = ov_FLAGS.count(True) #How many sides
    
        #If overflow on single side only
        if ncomp == 1:
            comp_dim = ov_FLAGS.index(True)
            comp_dim_ = (comp_dim-2) if comp_dim > 1 else (comp_dim+2) #Find the opposing dim to add gap to
            comp = abs(ov[comp_dim])

            output[comp_dim] = 1 if comp_dim in (2,3) else 0 #Check which opposing side of a dim it is
            output[comp_dim_] = (output[comp_dim_]-comp) if comp_dim in (2,3) else (output[comp_dim_]+comp)

            return output

        #If overflow on more than one side.
        if ncomp > 1:

            #Check which sides spills
            comp_dims = []
            for i,j in enumerate(ov_FLAGS):
                if j is True:
                    comp_dims.append(i)
             
            #If spill over both side of a single dimension
            if (0 in comp_dims and 2 in comp_dims) or (1 in comp_dims and 3 in comp_dims):
                raise Exception("Not implemented since this does not happen for the VisDrone2018-SOT dataset.")
                
            #If spill over in sides of different dim
            else:
                comp_dim1 = comp_dims[0]
                comp_dim2 = comp_dims[1]
                comp_dim1_ = (comp_dim1-2) if comp_dim1 > 1 else (comp_dim1+2) #Find the opposing dim to add gap to
                comp_dim2_ = (comp_dim2-2) if comp_dim2 > 1 else (comp_dim2+2) #Find the opposing dim to add gap to
                comp1 = abs(ov[comp_dim1])
                comp2 = abs(ov[comp_dim2])

                output[comp_dim1] = 1 if comp_dim1 in (2,3) else 0 #Check which opposing side of a dim it is
                output[comp_dim1_] = (output[comp_dim1_]-comp1) if comp_dim1 in (2,3) else (output[comp_dim1_]+comp1)
                output[comp_dim2] = 1 if comp_dim2 in (2,3) else 0 #Check which opposing side of a dim it is
                output[comp_dim2_] = (output[comp_dim2_]-comp2) if comp_dim2 in (2,3) else (output[comp_dim2_]+comp2)    

                return output


    else: #If no need comp
        return output


def compute_excc(nex_tw,nex_th,nex_bw,nex_bh,req):
    """
    Computes the spacing on each side of an exemplar for cropping. 
    Returns normalized coordinates to crop with.

    If overflows happens in two sides of a same dimension (e.g. scene size req is larger than entire full scene)
    the function will return the largest square image possible covering the exemplar. Make sure to have a resize
    catching such cases on the return of this function.
    """

    scene_w, scene_h = req[0], req[1]

    #Compute exem dim
    exem_w = nex_bw - nex_tw
    exem_h = nex_bh - nex_th
        
    #Catch problematic inputs
    assert req[0] <= 1, "Error! Patch size asked for is bigger than the actual pic. req[0]: {}".format(req[0])
    assert req[1] <= 1, "Error! Patch size asked for is bigger than the actual pic. req[1]: {}".format(req[1])

    #Size the gap needed
    req_w = scene_w - exem_w
    req_h = scene_h - exem_h

    #Randomize translation
    spf1 = random.uniform(0,1) #Split factor
    req_w1 = req_w * spf1
    req_w2 = req_w - req_w1

    spf2 = random.uniform(0,1)
    req_h1 = req_h * spf2
    req_h2 = req_h - req_h1

    #Check which side overflows
    ov_left = True if nex_tw < req_w1 else False 
    ov_right = True if (nex_bw + req_w2) > 1 else False
    ov_top = True if nex_th < req_h1 else False
    ov_bottom = True if (nex_bh + req_h2) > 1 else False

    ov_FLAGS = [ov_left,ov_top,ov_right,ov_bottom]
    ov = [req_w1-nex_tw, req_h1-nex_th, (nex_bw + req_w2)-1,(nex_bh + req_h2)-1] #How much spill over

    need_comp = True if any(ov_FLAGS) else False
    
    #Default cropping with no spillage
    new_th = nex_th - (req_h1)
    new_bh = nex_bh + (req_h2)
    new_tw = nex_tw - (req_w1)
    new_bw = nex_bw + (req_w2)
    output = [new_tw,new_th,new_bw,new_bh]

    #Comp needed
    if need_comp:
        ncomp = ov_FLAGS.count(True) #How many sides
    
        #If overflow on single side only
        if ncomp == 1:
            comp_dim = ov_FLAGS.index(True)
            comp_dim_ = (comp_dim-2) if comp_dim > 1 else (comp_dim+2) #Find the opposing dim to add gap to
            comp = abs(ov[comp_dim])

            output[comp_dim] = 1 if comp_dim in (2,3) else 0 #Check which opposing side of a dim it is
            output[comp_dim_] = (output[comp_dim_]-comp) if comp_dim in (2,3) else (output[comp_dim_]+comp)

            return output

        #If overflow on more than one side.
        if ncomp > 1:

            #Check which sides spills
            comp_dims = []
            for i,j in enumerate(ov_FLAGS):
                if j is True:
                    comp_dims.append(i)
             
            #If spill over both side of a single dimension
            if (0 in comp_dims and 2 in comp_dims) or (1 in comp_dims and 3 in comp_dims):
                raise Exception("Not implemented since this does not happen for the VisDrone2018-SOT dataset.")
                
            #If spill over in sides of different dim
            else:
                comp_dim1 = comp_dims[0]
                comp_dim2 = comp_dims[1]
                comp_dim1_ = (comp_dim1-2) if comp_dim1 > 1 else (comp_dim1+2) #Find the opposing dim to add gap to
                comp_dim2_ = (comp_dim2-2) if comp_dim2 > 1 else (comp_dim2+2) #Find the opposing dim to add gap to
                comp1 = abs(ov[comp_dim1])
                comp2 = abs(ov[comp_dim2])

                output[comp_dim1] = 1 if comp_dim1 in (2,3) else 0 #Check which opposing side of a dim it is
                output[comp_dim1_] = (output[comp_dim1_]-comp1) if comp_dim1 in (2,3) else (output[comp_dim1_]+comp1)
                output[comp_dim2] = 1 if comp_dim2 in (2,3) else 0 #Check which opposing side of a dim it is
                output[comp_dim2_] = (output[comp_dim2_]-comp2) if comp_dim2 in (2,3) else (output[comp_dim2_]+comp2)    

                return output


    else: #If no need comp
        return output

    
    
    

def fetch_exem(img_dir,full_imgs,ex_int,annotls,percent_neg=.5):
    """
    Generates exem dynamically. Grab 4 exem frames preceeding the current frame at a given interval.
    Exems taken with a square crop and resized to 224x224.
    
    Capable of returning a given percentage of postive/negative exem samples wrt percent_neg.

    [Need optimization and cleaning].
    """
    exem_dim = 224
    
    #Parse dir
    name,suffix = img_dir.stem, img_dir.suffix
    
    buffer_size = 4 
    curr_idx = parse_idx(name) + 1 #Compensate for parse_idx which is used to find annotations for an image and hence is 1-indexed.
    to_fetch = []
    imgs_buffer = []
    
    #Check posneg
    neg_req = round(percent_neg*buffer_size)
    pos_req = buffer_size - neg_req
    posneg_FLAG = [] #Need to ensure percent negatives are enforced
    
    assert neg_req + pos_req == buffer_size, "Error in fetch_exem posneg buffer computation!"
    
    #Gen exem idx to fetch
    while len(to_fetch) < buffer_size:
        n_name = namify(curr_idx)
        fname = n_name + suffix
        exem_dir = Path(str(img_dir.parent) + '/' + fname)
        
        #Validate if imgs is present in dataset.
        try:
            _ = PIL.Image.open(exem_dir)
            
        except:
            
            #If it fails. Wiggle curr_idx and continue.
            if curr_idx == 0:
                curr_idx += 5 #Just picked 5 so as to not overlap with -1.
            
            else:
                curr_idx -= 1
                continue
        
        #If it passes, append and update curr_idx.
        to_fetch.append(curr_idx)
        curr_idx = max(0,curr_idx - ex_int)
        

    for n in to_fetch:
        n_name = namify(n)
        fname = n_name + suffix
        exem_dir = Path(str(img_dir.parent) + '/' + fname)
    
        #Annotation to crop exem from. Hardcode ilsvr ant dir format
        annot_base = Path(*annotls.parts[:-4]) 
        annot_dir = annot_base/Path(*exem_dir.parts[-4:-1])/Path(n_name+'.xml')
        annot = to_visfmt(parse_xml(annot_dir)[0][0])
        posneg = 1 if any(annot) == True else 0 #For vanilla neg sample, annots are all 0
        img_ = PIL.Image.open(exem_dir)
                
                
        if neg_req > 0 and posneg == 0:

            #Open image and adjust bbox. Then crop, resize and cache.
            img_,comp_annot = to_square_neg(annot,img_,exem_dim)

            imgs_buffer.append(img_)
            neg_req -= 1
            posneg_FLAG.append(0)
            
                
        elif pos_req > 0 and posneg == 1:
            #Open image and adjust bbox. Then crop, resize and cache.
            img_,comp_annot = to_square(annot,img_)

            imgs_buffer.append(img_)
            pos_req -= 1  
            posneg_FLAG.append(1)
            
                
        elif neg_req > 0 and pos_req == 0 and posneg == 1:
            #Open image and adjust bbox. Then crop, resize and cache.
            img_,comp_annot = to_square_neg(annot,img_,exem_dim)

            imgs_buffer.append(img_)
            neg_req -= 1
            posneg_FLAG.append(0)

        
        elif pos_req > 0 and neg_req == 0 and posneg == 0:
            
            #If it encounters a vanilla neg sample (no target in scene) just replicate from previous exem
            if len(imgs_buffer) > 0:
                
                #May have no pos in FLAG thus exception may be raised in .index(1) below.
                try:
                    idx_to_comp = posneg_FLAG.index(1)
                    imgs_buffer.append(imgs_buffer[idx_to_comp])
                    pos_req -= 1
                    posneg_FLAG.append(1)
            
                except:
                    pass
                
            elif len(imgs_buffer) == 0:
                pass
                #No pos_req is deducted here so it'll just loop and find one later.
        
        else:
            raise Exception('Encountered an unforseen and unimplemented check for posneg in fetch_exem.')
        
        
    #Check if we're short on exems for rare case where first exem is vanilla negative.
    if len(imgs_buffer) !=4:
        
        #Check if it's all empty or pos_req > 0 and no pos in posneg_FLAG
        if not posneg_FLAG or (pos_req > 0 and 1 not in posneg_FLAG):
            
            #Manually seek pos sample in sequence and call fetch_exem again.
            manual_img_dir = greedy_posseek(img_dir,full_imgs)
            imgs_buffer = fetch_exem(manual_img_dir,full_imgs,ex_int,annotls,percent_neg)
        
        
        else:
            #If there's sufficient sample in buffer
            while len(imgs_buffer) < 4:
                posneg_ = 1 if pos_req != 0 else 0
                posneg_ = 0 if neg_req != 0 else 1
                idx_to_comp = posneg_FLAG.index(posneg_)
                imgs_buffer.append(imgs_buffer[idx_to_comp])

                #Compensate indices
                if posneg_ ==1:
                    pos_req -= 1
                elif posneg_ ==0:
                    neg_req -= 1

        
    return imgs_buffer



def greedy_posseek(neg_img,img_ls):
    """
    Helper function that takes an negative sample and try to find a positive one in the same sequence, 
    obtained by greedily searching through the list of images.
    """
    
    seq = neg_img.parts[-2] #Seq dir
    
    #Search
    for i in img_ls:
        stat = i[0] #pos/neg
        img_dir = str(i[1])
        
        if stat == 1 and seq in img_dir:
            target = i[1]
            break
            
            
    return target




def namify(idx):
    """
    Helper function that pads a given file number and return it as per the dataset image name format.
    """
    len_data = 6 #Ilsvr images are in the form of 000000.JPEG
    len_ = len(str(idx))
    need = len_data - len_

    assert len_data >= len_, "Error! Image idx being fetched is incorrect. Invalid value."

    pad = '0'*need

    return pad+str(idx) 



def parse_idx(img_name):
    """
    Simple helper function that takes an image name and return the index position of the image.
    """
    bk = 0

    #Find where the significant digit appears
    prefix = img_name.split('.')[0][3:]

    for idx,alpha in enumerate(prefix):
        if int(alpha) == 0:
            continue
        else:
            bk = idx
            break

    num = int(prefix[bk:]) - 1 #Since image names start from 1

    return num


def parse_ant(ant):
    """
    Helper function used to parse the labels returned by dataloader (stringified).
    Returns a list of float.
    """
    parsed = []
    
    for a in ant:
        i = a.strip('()').split(',')
        i = [float(j) for j in i]
        parsed.append(i)
        
    return torch.tensor(parsed)



def parse_xml(path,args=None):
    orig_shape = None
    new_shape = None

    if args is not None:
        orig_shape = args[0]
        new_shape = args[1]

    bboxes = []
    track_id = 0
    occ = 0
    w,h = 0,0
    
    tree = ET.parse(path)
    root = tree.getroot()

    if root.findall('object'):
        for obj in root.findall('object'):
            #Read w-h
            track_id = float(obj.find('trackid').text)
            occ = float(obj.find('occluded').text)
            w = float(root.find('size').find('width').text)
            h = float(root.find('size').find('height').text)

            # Read the bbox
            bbox = obj.find('bndbox')
            x_left = float(bbox.find('xmin').text)
            y_top = float(bbox.find('ymin').text)
            x_right = float(bbox.find('xmax').text)
            y_bottom = float(bbox.find('ymax').text)

            if orig_shape is not None and new_shape is not None:
                x_left = x_left*new_shape[1]/orig_shape[1]
                y_top = y_top*new_shape[0]/orig_shape[0]
                x_right = x_right*new_shape[1]/orig_shape[1]
                y_bottom = y_bottom*new_shape[0]/orig_shape[0]

            bbox = [int(x_left),int(y_top),int(x_right),int(y_bottom)]
            bboxes.append(bbox)
            
    else:
        bboxes = [[0]]


    return(bboxes,track_id,occ,w,h)
    
    

def to_visfmt(annot):
    """
    Helper function that changes (tw,th,bw,bh) -> (tw,th,w,h).
    Used to convert annotations from ilsvr to visdrone's since most scripts are already
    written in visdrone's format.
    """
    
    #Check if it's an empty bbox (neg scene)
    if len(annot) > 1:
        load = [annot[0], annot[1], annot[2]-annot[0], annot[3]-annot[1]]
    else:
        load = [0,0,0,0]
        
    return load



def to_square(annot,img):
    """
    Helper function that takes in a target's bbox and convert it to a larger bbox that is square.
    Compensated annot is not needed for exemplars, but implemented as an extra.
    Used for exems.
    """
    
    #Check needed dim
    need = max(annot[2],annot[3]) #See if it's taller or wider
    img_sz = min(img.size)

    #Compute center
    cw = annot[0] + (annot[2]/2)
    ch = annot[1] + (annot[3]/2)
    
    #Normalize annot
    img_w, img_h = img.size[0], img.size[1]    
    annot = [annot[0],annot[1],annot[2]+annot[0],annot[3]+annot[1]] #Format needed by compute_cc
    annot_norm = [annot[0]/img_w,annot[1]/img_h,annot[2]/img_w,annot[3]/img_h]
    
    #If the req bbox to be square is > than img_sz
    if need > img_sz:        
        #Squash it as little as possible by making it as square as possible
        req = (img_sz/img_w, img_sz/img_h)
        tw_n,th_n,bw_n,bh_n = compute_excc(*annot_norm,req) #Key step; compute comp        
        
    #If it's within the image
    else:
        #Compute compensation when needed 
        req = (need/img_w, need/img_h)
        tw_n,th_n,bw_n,bh_n = compute_excc(*annot_norm,req) #Key step; compute comp
        
        
    #Unnormalize
    tw = tw_n * img_w
    th = th_n * img_h
    bw = bw_n * img_w
    bh = bh_n * img_h

    #Crop and resize
    cropped = img.crop((annot[0],annot[1],annot[2],annot[3])) #Stretch
    # cropped = img.crop((tw,th,bw,bh)) #No stretch
    rsz_fc1 = cropped.size[0]/224
    rsz_fc2 = cropped.size[1]/224
    cropped = cropped.resize((224,224),resample=PIL.Image.BICUBIC) #Resize
    
    #Compensate annotations. Clip vals.
    ant_tw = max(0,annot[0] - tw)
    ant_th = max(0,annot[1] - th)
    ant_bw = max(0,(bw-tw)-(bw - annot[2])) #basically, scene_w(or h) - gap_scene_exem = exem_anot
    ant_bh = max(0,(bh-th)-(bh - annot[3]))  
    
    #Compensate annotations. Resize val.
    ant_tw,ant_bw = ant_tw/rsz_fc1, ant_bw/rsz_fc1
    ant_th,ant_bh = ant_th/rsz_fc2, ant_bh/rsz_fc2 
    compensated_ant = [i if i <= 224 else 224 for i in [ant_tw,ant_th,ant_bw,ant_bh]] #Only clip max after scaling
        
        
    return cropped, compensated_ant
        

    
def to_square_neg(annot,img,size=224):
    """
    Helper function that takes in a target's bbox and convert it to a larger bbox that is square.
    Compensated annot is not needed for exemplars, but implemented as an extra.
    Used for negative exems and scenes.
    """
        
    #Conv annot from (tw,th,w,h) -> (tw,th,bw,bh)
    annot_ = [annot[0],annot[1],annot[2]+annot[0],annot[3]+annot[1]]
    
    #Check for empty spaces surrounding annot
    w,h = img.size[0],img.size[1]
    rw = w - annot_[2]
    bh = h - annot_[3]
    lw = annot_[0] - 0
    th = annot_[1] - 0
    
    #Det which space to take
    spaces = [lw,th,rw,bh]
    gap1 = max(spaces)
    gap1_idx = spaces.index(gap1)
    
    #Check counter dim
    if gap1_idx in [0,2]:
        gap2 = h #It's w, check h.
    
    elif gap1_idx in [1,3]:
        gap2 = w #It's h, check w.
    
    #Compute neg exem bbox
    gap_min = min(gap1,gap2)
    gap_min_idx = [gap1,gap2].index(gap_min)
    
    #If it fits
    if gap_min > size:
        
        gap_gap = gap_min - size
        begin = random.randint(0,gap_gap) #Pick random point to start exem square
        end = begin + size    
    
    #If it doesn't fit in 
    else:
        gap_gap = 0
        begin = random.randint(0,gap_gap) #Pick random point to start exem square
        end = begin + gap_min
        
    
    #Compensate absolute bbox val
    if gap1_idx == 0 or gap1_idx == 1:

        #Gap at left of bbox
        crop_ant = [begin,begin,end,end]

    elif gap1_idx == 2:

        #Gap at right of bbox
        crop_ant = [annot_[2]+begin,begin,annot_[2]+end,end]        

    elif gap1_idx == 3:

        #Gap at bottom of bbox
        crop_ant = [begin,annot_[3]+begin,end,annot_[3]+end]
        
        
    #Crop img and resize if needed
    cropped = img.crop(crop_ant)
    
    if cropped.size[0] != size or cropped.size[1] != size:
        cropped = cropped.resize((size,size),resample=PIL.Image.BICUBIC) #Resize
    
    compensated_ant = [0,0,0,0] #No target in negative sample
    
    
    return cropped, compensated_ant

    
    
def to_square_scene(annot,img):
    """
    Helper function that takes in a target's bbox and convert it to a larger bbox that is square.    
    Used for scene.
    """
    
    #Check needed dim
    need = 360 #Patch size is fixed at 360x360
    img_sz = min(img.size)

    #Normalize annot
    img_w, img_h = img.size[0], img.size[1]    
    annot = [annot[0],annot[1],annot[2]+annot[0],annot[3]+annot[1]] #Format needed by compute_cc
    annot_norm = [annot[0]/img_w,annot[1]/img_h,annot[2]/img_w,annot[3]/img_h]
    
    #If the req bbox to be square is > than img_sz
    if need > img_sz:
        #Squash it as little as possible by making it as square as possible
        req = (img_sz/img_w, img_sz/img_h)
        tw_n,th_n,bw_n,bh_n = compute_excc(*annot_norm,req) #Key step; compute comp        
        
    #If it's within the image
    else:
        #Compute compensation when needed 
        req = (need/img_w, need/img_h)
        tw_n,th_n,bw_n,bh_n = compute_excc(*annot_norm,req) #Key step; compute comp
        
    #Unnormalize
    tw = tw_n * img_w
    th = th_n * img_h
    bw = bw_n * img_w
    bh = bh_n * img_h

    #Crop and resize
    cropped = img.crop((tw,th,bw,bh))
    rsz_fc1 = cropped.size[0]/224
    rsz_fc2 = cropped.size[1]/224
    cropped = cropped.resize((224,224),resample=PIL.Image.BICUBIC) #Resize

    #Compensate annotations. Clip vals.
    ant_tw = max(0,annot[0] - tw)
    ant_th = max(0,annot[1] - th)
    ant_bw = max(0,(bw-tw)-(bw - annot[2])) #basically, scene_w(or h) - gap_scene_exem = exem_anot
    ant_bh = max(0,(bh-th)-(bh - annot[3]))  
    
    #Compensate annotations. Resize val.
    ant_tw,ant_bw = ant_tw/rsz_fc1, ant_bw/rsz_fc1
    ant_th,ant_bh = ant_th/rsz_fc2, ant_bh/rsz_fc2 
    compensated_ant = [i if i <= 224 else 224 for i in [ant_tw,ant_th,ant_bw,ant_bh]] #Only clip max after scaling
        
    return cropped, compensated_ant
        


def dt_trans(trans,scene,exems,buffer_size):
    """
    Function to enclose the transformation sequence used in dataset.
    """

    norm_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    #If transforms
    if trans:
        #Make sure trans applied on exem and scene are similar
        vals = [random.uniform(0.85,1.15) for i in range(4)]
        val_hue = random.uniform(-0.1,0.1)
        scene = transforms.functional.adjust_brightness(scene,vals[0])
        scene = transforms.functional.adjust_contrast(scene,vals[1])
        scene = transforms.functional.adjust_gamma(scene,vals[2])
        scene = transforms.functional.adjust_saturation(scene,vals[3])
        scene = transforms.functional.adjust_hue(scene,val_hue)
        scene = norm_trans(scene)

        for i,exem_ in enumerate(exems):
            exem_ = transforms.functional.adjust_brightness(exem_,vals[0])
            exem_ = transforms.functional.adjust_contrast(exem_,vals[1])
            exem_ = transforms.functional.adjust_gamma(exem_,vals[2])
            exem_ = transforms.functional.adjust_saturation(exem_,vals[3])
            exem_ = transforms.functional.adjust_hue(exem_,val_hue)
            exem_ = norm_trans(exem_)
            exems[i] = exem_
        exems = torch.stack(exems)

    else:
        scene = norm_trans(scene)
        exems = [norm_trans(exem_) for exem_ in exems]
        exems = torch.stack(exems)

    return scene,exems




class gen_dt(Dataset):

    def __init__(self, img_root, ant_root, posneg_ls, pos_ls, neg_ls, cusneg_ls, 
                 percent_neg = 0,ex_int = 16, transform = True):
        """
        General buffer dataset class.

        Rather than taking in img_dir, feed in instead a csv that can be decoded to return
        the paths for the negative and positive samples respectively.Instantiate different 
        dataset for train and valid.The annotations for the entire train/valid set will be 
        loaded on instantiation to prevent read/write at every sample.

        Input:
            img_root: Root to ImageNet-VID.
            ant_root: Root to annotations of corresponding img_root.
            posneg_ls: Path to posneg.csv. Determines the positivity-negativity of sample.
            pos_ls: Path to ilsvrc_vanilla_pos.csv. Used to fetch positive samples as dictated by posneg_ls.
            neg_ls: Path to ilsvrc_vanilla_neg.csv. Used to fetch negative samples as dictated by posneg_ls.
            cusneg_ls: Path to neg_below08.csv. Used to generate custom negative samples.
            percent_neg: Valid values are [0,0.25,0.5,0.75]. Used to control percentage of distractor exemplars in buffer.
            ex_int: Time interval (frames) between exemplars in buffer.
            transform: Should always be TRUE in usage. Used to transform images for PyTorch.

        Return:
            Img/scene: A portion of the scene with/without a target. Size of [3x224x224]. 
            Img/exems: Collection of exemplars taken from the same sequence as the scene. Size of [buffer_sizex3x224x224].
            Img/pth_full: Full path for the scene.
            Img/seq_name: Name of sequence for the scene.
            Img/img_name: Name of image for the scene.
            Annot/bbox: Ground truth bounding box coordinates. Quick-fixed by stringify to prevent jumbling.
            Annot/obj: Ground truth objectness score of scene. Quick-fixed by stringify to prevent jumbling
        """

        #Setup
        self.ex_int = ex_int
        self.transform = transform
        self.posneg_ls = posneg_ls
        self.pos_ls = pos_ls
        self.neg_ls = neg_ls
        self.cusneg_ls = cusneg_ls
        self.img_root = img_root
        self.ant_root = ant_root
        self.percent_neg = percent_neg     
        self.transform = transform
        self.data, self.annot = compile_data(self.img_root,self.ant_root,self.posneg_ls,
                                             self.pos_ls,self.neg_ls,self.cusneg_ls)
    
        assert len(self.annot) == len(self.data), "Error! The len(annot) != len(imgs)"

        self.data, self.annot = np.array(self.data),np.array(self.annot)

        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        ###Parse path
        img_full = self.data[idx][1] #self.data is a list in form [[pos/neg,path],...]
        img_name = img_full.parts[-1]
        seq_name = img_full.parts[-2]
        
        ###Open Image
        full_scene = PIL.Image.open(img_full)
        
        ###Fetch label. 
        #Parse_xml returns extra info. (w,h,occ,extra bboxes if >1 target)
        #self.annot is a list in form [[pos/neg,path],...] 
        annot = parse_xml(self.annot[idx][1])[0][0]
               
        #convert (tw,th,bw,bh) -> (tw,th,w,h) ; (ilsvr) -> (visdrone) default format
        annot = to_visfmt(annot)
        
        ###Fetch Positive or Negative sample
        stat = self.data[idx][0]
        assert stat == self.annot[idx][0], "Error! Both img and ant should be equal in pos/neg, not diff."
        
        #If it's a positive sample
        if stat == 1:
            
            #Fetch, crop and transform scene with compensated annot
            scene,annot = scene_crop(full_scene,annot)    
                
        #If it's a negative sample
        elif stat == 0:

            #Fetch, crop and transform scene with compensated annot
            scene,annot = scene_crop_neg(full_scene,annot)  
            
        else:
            raise Exception("Error! Invalid stat value")

            
        #Fetch exemplars
        exems = fetch_exem(img_full,self.data,self.ex_int,self.annot[0][1],self.percent_neg)
           
        ###Transforms
        scene,exems = dt_trans(self.transform,scene,exems,4)
            
            
        ###Package
        load = {"Img":{"scene":scene,
                        "exem":exems,
                        "pth_full":str(img_full),
                        "seq_name":str(seq_name),
                        "img_name":str(img_name)},
                "Annot":{"bbox":str(annot),
                         "obj":str(stat)}}
        
        
        return load
        
