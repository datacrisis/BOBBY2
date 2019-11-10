import matplotlib.pyplot as plt
import time,os,copy, argparse, cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
from net import BOBBY2


#Setup interface. [Unfinished Impe
parser = argparse.ArgumentParser(description='BOBBY2 Tracking Demo on GOT-10k')
parser.add_argument('-m','--model', default='../weights/bobby2.pth',help='path to trained model')
args = parser.parse_args()

#Tracker class
class BBY2_Tracker(object):
    
    def __init__(self):
        
        #Init tracker
        self.count = 0
        self.redet_thres = 3
        self.reset_thres = 1
        self.skip_frame = 100e100
        self.window_fc = 1.2
        self.init_window_fc = self.window_fc
        self.ex_int = 150
        self.bmax = 4
        self.conf_fc = 2
        self.thres1, self.thres2 = 0.5,0.5
        self.action_size = 6
        self.bby2 = BOBBY2(1,self.action_size)
        self.bby2.load_state_dict(torch.load(args.model,map_location=lambda storage, loc: storage))
        self.bby2.eval()
        self.init_ex_int = [0.5]
        self.obj_hist = []

        #Init utility. To Cuda and normalizing
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
        
    
    def init(self):
        """
        [Docstring to be udpated]
        Initialize your tracking model in the first frame
        Implemented in to forward pass over entire exemplar buffer in one-shot for simplicity.
        """
        
        #Init stats
        image, exem, box = self.init_frame()
        self.box = box
        self.bbox = self.box
        tw,th,w,h = [int(i) for i in self.box]
        bw,bh = tw+w, th+h
        self.window = max(w,h) * self.window_fc #Basically taking 2 times exem as scene

        # #Computing BBOX.
        # left = max(tw, 0)
        # top = max(th, 0)
        # right = min(tw + w, image.size[1] - 1)
        # bottom = min(th + h, image.size[0] - 1)
        
        #Computing center and size
        self.center = (tw + (w / 2), th + (h / 2))
        self.size = (w, h)
        
        #Init buffer and cut exemplar
        self.template_1 = exem.resize((224,224),resample=Image.NEAREST)        
        self.buff_payload = []
        
        for i in range(self.bmax):
            self.buff_payload.append(self.template_1)
        
        #Store them for later refreshing
        self.b1 = self.template_1
        self.b2 = self.template_1
        self.b3 = self.template_1
        self.b4 = self.template_1    
        
        #Transform
        self.buff_payload_1 = [self.norm(exem_) for exem_ in self.buff_payload]
        self.buff_payload_1 = torch.stack(self.buff_payload_1)
        
        #Featurize and store
        out = self.bby2(0,self.buff_payload_1,True)
        self.buffer = torch.cat((out[0],out[1],out[2],out[3])).unsqueeze(0)
                
        return image

        
    def update(self, image):
        """
        [Docstring to be udpated]
        Track in current frame and fetch next frame.            
        """

        #Drop frame if needed.
        if self.count % self.skip_frame == 0 and self.count != 0 and self.count % self.ex_int != 0:
            
            self.count += 1
            return self.bbox    


        #Check for buffer refresh. Don't take as exem if obj != 1
        if (self.count % self.ex_int == 0 and self.count != 0 and self.objectness) or (self.count in self.init_ex_int and self.objectness):
            
            #Update template
            # self.b2 = self.b1
            # self.b3 = self.b2
            # self.b4 = self.b3
            self.b1 = self.scene_raw.crop((int(self.out_bbox[0]),
                                                 int(self.out_bbox[1]),
                                                 int(self.out_bbox[2]),
                                                 int(self.out_bbox[3]))).resize((224,224),resample=Image.NEAREST)
            
            self.buff_payload = [self.norm(self.b1),self.norm(self.b2),self.norm(self.b3),self.norm(self.b4)]            
            self.buff_payload_1 = torch.stack(self.buff_payload)

            #Featurize and store
            out = self.bby2(0,self.buff_payload_1,True)
            self.buffer = torch.cat((out[0],out[1],out[2],out[3])).unsqueeze(0)
            
            
            #Drop frame for this turn since we're refreshing exem. Comment the two lines below out to not drop on refresh.
            self.count += 1
            return self.bbox
        
        
        #Computing scene coordinates
        self.left = max(int(self.center[0] - float(self.window) / 2), 0)
        self.top = max(int(self.center[1] - float(self.window) / 2), 0)
        self.right = min(int(self.center[0] + float(self.window) / 2), image.size[0] - 1)
        self.bottom = min(int(self.center[1] + float(self.window) / 2), image.size[1] - 1)
        
        #Compute resize factor
        scene_w = self.right - self.left
        scene_h = self.bottom - self.top
        rsz_fc = (scene_w/224,scene_h/224)
        
        #Crop
        self.scene_raw = image.crop((self.left,self.top,self.right,self.bottom)).resize((224,224),resample=Image.NEAREST)
        self.scene = self.norm(self.scene_raw).unsqueeze(0)
        
        #Track
        out = self.bby2(self.scene,self.buffer,False)
        self.out_bbox = out[:,:4][0]
        self.out_obj = torch.nn.functional.softmax(out[:,4:][0]) #Squeeze it
        
        #Parse output
        out_bbox = [int(i) for i in self.out_bbox]
        tw_,th_,bw_,bh_ = out_bbox
        self.objectness = 1 if (self.out_obj[1]/self.out_obj[0]) > self.conf_fc else 0 
        self.obj_hist.append(self.objectness)


        #Resize output back to scene dim
        tw,th,bw,bh = tw_ * rsz_fc[0], th_ * rsz_fc[1], bw_ * rsz_fc[0] , bh_ * rsz_fc[1]
        w = bw - tw
        h = bh - th

        #Dampening width and height. Only update if valid.
        if abs(w-self.size[0])/(self.size[0]+0.001) < self.thres1 and abs(h-self.size[1])/(self.size[1]+0.001) < self.thres2:

            #Refresh infos
            self.window = max(w, h) * self.window_fc
            self.size = (w,h)
            self.center = (self.left + tw + (float(self.size[0]) / 2), self.top + th + (float(self.size[1]) / 2))
            self.out_bbox = out_bbox

            # Final box. Update only we're confident that there's an object.
            if self.objectness:
                self.bbox = [self.left+tw, self.top+th, w, h]

        #Update count
        self.count += 1
        return self.bbox


    def track(self):
        """
        [Docstring to be updated]
        Function to track.
        """
        #Init image
        image = self.init()

        #Setup cam
        cam = cv2.VideoCapture(0)
        count = 0

        #Track
        while True:
            
            #Update tracker
            self.bbox = self.update(image)

            #Draw box
            self.bbox = [int(j) for j in self.bbox]
            drawn = np.array(image)[:, :, ::-1].copy()
            drawn = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)
            cv2.rectangle(drawn, (self.bbox[0], self.bbox[1]), (self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3]), (255,0,0), 2)

            #Display
            cv2.imshow('Tracking Output', drawn)

            #Uncomment to write out
            # cv2.imwrite('/path/to/save/{}.png'.format(count),drawn)
            # count += 1

            #To escape
            if cv2.waitKey(1) == 27: 
                break  # esc to quit

            #Capture
            _, image = cam.read()
            image = Image.fromarray(image)

        #Close windows
        cv2.destroyAllWindows()

        return None


    def init_frame(self):
        """
        Function to capture cam footage.
        """
        #Init params
        sz_w,sz_h = 250,250

        #Capture
        cam = cv2.VideoCapture(0)
        while True:
            _, img = cam.read()

            #Draw box
            w,h = img.shape[1],img.shape[0]
            cw,ch = (w/2),(h/2)
            tw,th = int(cw - (sz_w/2)), int(ch - (sz_h/2))
            bw,bh = int(cw + (sz_w/2)), int(ch + (sz_h/2))
            to_draw = img.copy()
            cv2.rectangle(to_draw, (tw, th), (bw, bh), (255,0,0), 2)

            #Display
            cv2.imshow('Your Face Here', to_draw)

            #To escape
            if cv2.waitKey(1) == 27: 
                break  # esc to quit

        #Close windows
        cv2.destroyAllWindows()

        #Transform to suitable colorspace
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        coord = (tw,th,bw,bh)

        #Save last frame for ref. Uncomment to save.
        # cv2.imwrite('/path/to/save/first.png',to_draw)

        #Crop exem
        exem = img[coord[1]:coord[3],coord[0]:coord[2],:]
        
        #Convert to PIL format for tracker. Quick fix :D.
        img = Image.fromarray(img)
        exem = Image.fromarray(exem)
        
        return img, exem, coord




if __name__ == '__main__':
    tracker = BBY2_Tracker()
    _ = tracker.track()
    
