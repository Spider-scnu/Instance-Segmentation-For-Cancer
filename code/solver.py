import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import scipy.misc
from PIL import Image
import shutil
def saveImg(raw, hw, threshold = 0.5):
    #print(raw)
    raw = raw > threshold
    imgRevert = raw  * 255.0
    #imgRevert = raw
    #print(imgRevert)
    imgRevert = imgRevert.cpu().detach().numpy()
    imgRevert = imgRevert[0,:,:,:]
    
    imgRevert = imgRevert.transpose(1,2,0)
    #print(np.max(imgRevert))
    #print(np.array(np.uint8(imgRevert[:,:,0])))
    imgRevert = np.repeat(imgRevert, 3, axis=2)
    im = Image.fromarray(np.uint8(imgRevert))
    
    im = im.resize((hw[0],hw[1]))
    return im

def saveImg_GT(raw, hw):
    imgRevert = raw * 255.0#(raw * 0.5 + 0.5) * 255.0
    imgRevert = imgRevert.cpu().numpy()
    imgRevert = imgRevert[0,:,:,:]
    
    imgRevert = imgRevert.transpose(1,2,0)
    #print(np.max(imgRevert))
    #print(np.array(np.uint8(imgRevert[:,:,0])))
    imgRevert = np.repeat(imgRevert, 3, axis=2)
    im = Image.fromarray(np.uint8(imgRevert))
    
    im = im.resize((hw[0],hw[1]))
    return im

def saveImg_contour(raw, hw):
    imgRevert = (raw * 0.5 + 0.5) * 255.0
    imgRevert = imgRevert.cpu().numpy()
    imgRevert = imgRevert[0,:,:,:]
    
    imgRevert = imgRevert.transpose(1,2,0)
    #print(np.max(imgRevert))
    #print(np.array(np.uint8(imgRevert[:,:,0])))
#     imgRevert = np.repeat(imgRevert, 3, axis=2)
    im = Image.fromarray(np.uint8(imgRevert))
    
    im = im.resize((hw[0],hw[1]))
    return im

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)


        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img


    def train(self, pretrain, pre_bestscore):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        if pretrain == 0:
            unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
        else:
            unet_path = self.model_path
        #print(unet_path)
        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
            # Train for Encoder
            lr = self.lr
            best_unet_score = pre_bestscore

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length = 0
                #print(self.train_loader)
                for i, (images, GT, _, _) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    #print(i, (images, GT))
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    #print(images.shape, GT.shape)
                    # SR : Segmentation Result
                    SR = self.unet(images)
                    #print(SR.shape)
                    SR_probs = F.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0),-1)

                    GT_flat = GT.view(GT.size(0),-1)
                    
                    loss = self.criterion(SR_flat,GT_flat)
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)
                    length += images.size(0)

                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                      epoch+1, self.num_epochs, \
                      epoch_loss,\
                      acc,SE,SP,PC,F1,JS,DC))



                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))


                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length=0
                for i, (images, GT, _, _) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)

                    length += images.size(0)

                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))

                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''


                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    premodel_unet_path = unet_path[:-4]+'_pretrained'+'.pkl'
                    print('Best %s model score : %.4f unet_path is '%(self.model_type,best_unet_score), premodel_unet_path)
                    torch.save(best_unet, premodel_unet_path)
                    
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length = 0
                #print(self.train_loader)
                for i, (images, GT, _, _) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    #print(i, (images, GT))
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR_probs = F.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0),-1)

                    GT_flat = GT.view(GT.size(0),-1)
                    loss = self.criterion(SR_flat,GT_flat)
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)
                    length += images.size(0)

                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                      epoch+1, self.num_epochs, \
                      epoch_loss,\
                      acc,SE,SP,PC,F1,JS,DC))



                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))


                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length=0
                for i, (images, GT, _, _) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)

                    length += images.size(0)

                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))

                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''


                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(best_unet,unet_path)

            #===================================== Test ====================================#
            del self.unet
            del best_unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))
            
            self.unet.train(False)
            self.unet.eval()

            acc = 0.	# Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length=0
            for i, (images, GT, _, _) in enumerate(self.valid_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images)
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)

                length += images.size(0)

            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC


            f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
            f.close()

    def test(self, unet_path, result_savepath, mask_savepath, pre_savepath, threshold=0.5):
        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        length=0
        num_recall = 0
        rm_mkdir(result_savepath)
        rm_mkdir(mask_savepath)
        rm_mkdir(pre_savepath)
        for i, (images, GT, HW, filename) in enumerate(self.test_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = self.unet(images)
            acc += get_accuracy(SR, GT, threshold)
            SE += get_sensitivity(SR, GT, threshold)
            SP += get_specificity(SR, GT, threshold)
            PC += get_precision(SR, GT, threshold)
            F1 += get_F1(SR, GT, threshold)
            JS += get_JS(SR, GT, threshold)
            DC += get_DC(SR, GT, threshold)
            
            GT_class = torch.max(GT).int()
            SR_class = torch.max(SR>threshold)
            GT_class = GT_class.type_as(SR_class)
            # recall positive 
            if SR_class>0:
                num_recall+=1
                #print(GT_class, SR_class)
                SR_PIL_img = saveImg(SR, HW)
                GT_PIL_img = saveImg_GT(GT, HW)
                images_PIL_img = saveImg_contour(images, HW)
                #filename = self.test_loader.dataset.image_paths[i].split('/')[-1][:-4]
                #print(filename[0])
                SR_PIL_img.save(pre_savepath+filename[0]+".png")
                GT_PIL_img.save(mask_savepath+filename[0]+"_mask.png")
                images_PIL_img.save(result_savepath+filename[0]+".png")
            length += images.size(0)
#             images = images.cpu().numpy()
            #new_img_PIL = torchvision.transforms.ToPILImage()(images[0,:,:,:]).convert('RGB')
#             scipy.misc.imsave('outfile.jpg', images)
            #SR_PIL_img = saveImg(SR, HW)

        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        unet_score = JS + DC
        recall = num_recall/length
        print('acc:{} DC:{} F1:{}'.format(acc, DC, F1))
        print('length:{} num_recall:{} recall:{}'.format(length, num_recall, recall))
    def predict(self, unet_path, raw_savepath, pre_savepath, zeros_savepath, threshold = 0.5):
        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        num_recall = 0
        length = 0
        rm_mkdir(raw_savepath)
        rm_mkdir(pre_savepath)
        rm_mkdir(zeros_savepath)
        zeors_image = np.zeros(shape=(512,512,3))
        for i, (images, GT, HW, filename) in enumerate(self.test_loader):

            images = images.to(self.device)
            SR = self.unet(images)
            
            SR_class = torch.max(SR>threshold)

            if SR_class>0:
                num_recall+=1
                images_PIL_img = saveImg_contour(images, HW)
                SR_PIL_img = saveImg(SR, HW)
                #filename = self.test_loader.dataset.image_paths[i].split('/')[-1][:-4]
                images_PIL_img.save(raw_savepath+filename[0]+".png")
                SR_PIL_img.save(pre_savepath+filename[0]+".png")
            else:
                scipy.misc.imsave(zeros_savepath+filename[0]+".png", zeors_image)
            length += images.size(0)
            #print(HW)

        recall = num_recall/length
        print('length:{} num_recall:{} recall:{}'.format(length, num_recall, recall))
        #print('acc:{} DC:{} F1:{} recall:{}'.format(acc, DC, F1, recall))


