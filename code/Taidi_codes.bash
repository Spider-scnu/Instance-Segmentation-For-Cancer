#### 打包代碼
zip -r ./R2AttU_net_result_DC0.55.zip ./R2AttU_Net_Img_Savepath


#### ISIC 2018 

#------------------R2U_net
    nohup python -u ../main.py --num_epochs=30 --num_epochs_decay=29 --model_type='R2U_Net' > log_R2U_Net_g3 &


#------------------R2AttU_net
    nohup python -u ../main.py --num_epochs=300 --num_epochs_decay=200 --best_score=1.27 --pretrained=1 --model_path='./models/R2AttU_Net-100-0.0000-78-0.4701_pretrained.pkl' --model_type='R2AttU_Net' > log_R2AttU_Net_preISIC2 &

#------------------AttU_net
    nohup python -u ../main.py --num_epochs=400 --num_epochs_decay=200 --model_type='AttU_Net' > log_AttU_Net &

    nohup python -u ../main.py --num_epochs=400 --num_epochs_decay=200 --best_score=0.7 --pretrained=1 --model_path='./models/AttU_Net-400-0.0004-233-0.4042.pkl' --model_type='AttU_Net' > log_AttU_Net_2 &

python ../main.py --mode='test' --model_type='AttU_Net' --testmodel_path='./models/AttU_Net-400-0.0004-233-0.4042.pkl'

##################################################################################################################



#### Cancer 2019

#------------------U_net
    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=70 --best_score=1.05 --pretrained=1 --model_type='U_Net' --model_path='./models/U_Net-100-0.0004-2-0.0003.pkl' > log_U_Net_pre_Cancer_g1 &

#------------------AttU_net
    #------------------ pre-trained
    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=200 --best_score=1.05 --pretrained=1 --model_type='AttU_Net' --model_path='./models/AttU_Net-400-0.0004-233-0.4042_pretrained.pkl' > log_AttU_Net_pre1_Cancer_g2 &
    
    nohup python -u ../main_cancer.py --num_epochs=200 --num_epochs_decay=180 --best_score=0.7 --pretrained=1 --model_path='./models/AttU_Net-400-0.0004-233-0.4042_pretrained.pkl' --model_type='AttU_Net' > log_cancer_AttU_Net_2_pre2 &
    
    #------------------ test
    python ../main_cancer.py --mode='test' --model_type='AttU_Net' --testmodel_path='./models/AttU_Net-400-0.0004-233-0.4042_pretrained_pretrained.pkl' --Img_savepath='./AttU_Net_Img_Savepath/'


#------------------R2AttU_net

   #------------------ trained 
   nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=200 --model_type='R2AttU_Net' > log_R2AttU_Net_Cancer_g3 &
    
   #------------------ pre-trained
    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=20 --best_score=1.3 --pretrained=1 --lr=0.01 --model_path='./models/R2AttU_Net-100-0.0000-78-0.4701_pretrained_pretrained.pkl' --model_type='R2AttU_Net' > log_R2AttU_Net_pre2_320_320_Cancer_g3 &
    
    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=70 --best_score=1.1 --pretrained=1 --model_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re.pkl' --model_type='R2AttU_Net' > log_R2AttU_Net_pre_320_320_Cancer_g2 &

    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=50 --best_score=1.29 --pretrained=1 --lr=0.0001 --model_path='./models/R2AttU_Net-100-0.0000-78-0.4701_pretrained_pretrained.pkl' --model_type='R2AttU_Net' > log_R2AttU_Net_pre4Cancer_g3 &


   #------------------ test
        python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_pretrained_pretrained.pkl' --Img_savepath='./R2AttU_Net_Img_Savepath/'
    
        #----------- test none images
        python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./Prediction/R2AttU_Net_Img_Savepath_none/' --Mask_savepath='./Prediction/R2AttU_Net_Img_Savepath_none/' --test_path='./dataset/testcancer_none/' --threshold=65
        
        #----------- test true images
        python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./Prediction/R2AttU_Net_Img_Savepath_trues/' --Mask_savepath='./Prediction/R2AttU_Net_Img_Savepath_trues/' --test_path='./dataset/testcancer_true/' --threshold=90
        
        python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./Prediction/R2AttU_Prediction_Img/' --Mask_savepath='./Prediction/R2AttU_Prediction_Mask/' --test_path='./dataset/testcancer_all/' --threshold=90
        
        python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./R2AttU_Net_Img_Savepath_true/'  --test_path='./dataset/testcancer_true/' 

#----------------------R2U_net

    #------------------ pre-trained
    nohup python -u ../main_cancer.py --num_epochs=300 --num_epochs_decay=70 --best_score=1.0 --pretrained=1 --model_path='./models/R2U_Net-30-0.0013-12-0.4422.pkl' --model_type='R2U_Net' > log_R2U_Net_pre1_Cancer_g3 &
    

# Masked R-CNN

    #----------------- thread
    export OMP_NUM_THREADS=1
    export USE_SIMPLE_THREADED_LEVEL3=1
    
    # ----------------- pretrain ISIC
    nohup python -u samples/balloon/balloon_ISIC.py train --dataset=./dataset/ISIC --weights=coco > log_MaskRCNN_320_320_ISIC_g2 &
    
    #----------------- train
    nohup python -u samples/balloon/balloon.py train --dataset=./dataset --weights=coco > log_MaskRCNN_320_320_cancer_g3_190426T2354 &
    nohup python -u samples/balloon/cancer.py train --dataset=./dataset --weights=coco > log_MaskRCNN_320_320_cancer_g3_190426T2354 &





### 生成数据集
    # ---------- 0 & 1
    python ../dataset_cancer.py --train_ratio=0.75 --valid_ratio=0.15 --test_ratio=0.1
    # ---------- 0
    python ../dataset_cancer.py --train_ratio=0.0\
                                --valid_ratio=0.0\
                                --test_ratio=1.0\
                                --origin_data_path='../ISIC/dataset/cancer/Input_all'\
                                --origin_GT_path='../ISIC/dataset/cancer/Mask_all'\
                                --train_path='./dataset/traincancer_all/'\
                                --train_GT_path='./dataset/traincancer_all_GT/'\
                                --valid_path='./dataset/validcancer_all/'\
                                --valid_GT_path='./dataset/validcancer_all_GT/'\
                                --test_path='./dataset/testcancer_all/'\
                                --test_GT_path='./dataset/testcancer_all_GT/'
    # ----------- 1
    python ../dataset_cancer.py --train_ratio=0.899\
                                --valid_ratio=0.1\
                                --test_ratio=0.001\
                                --origin_data_path='../ISIC/dataset/cancer/Input_true'\
                                --origin_GT_path='../ISIC/dataset/cancer/Mask_true'\
                                --train_path='./dataset/traincancer_true/'\
                                --train_GT_path='./dataset/traincancer_true_GT/'\
                                --valid_path='./dataset/validcancer_true/'\
                                --valid_GT_path='./dataset/validcancer_true_GT/'\
                                --test_path='./dataset/testcancer_true/'\
                                --test_GT_path='./dataset/testcancer_true_GT/'
                                
### 預測
    # -----------------
    python ../dataset_cancer.py --train_ratio=0.0\
                                --valid_ratio=0.0\
                                --test_ratio=1.0\
                                --origin_data_path='../ISIC/dataset/cancer/Input_pre'\
                                --origin_GT_path='../ISIC/dataset/cancer/Mask_pre'\
                                --train_path='./dataset/traincancer_pre/'\
                                --train_GT_path='./dataset/traincancer_pre_GT/'\
                                --valid_path='./dataset/validcancer_pre/'\
                                --valid_GT_path='./dataset/validcancer_pre_GT/'\
                                --test_path='./dataset/testcancer_pre/'\
                                --test_GT_path='./dataset/testcancer_pre_GT/'
                                
   # --------------------
   python ../main_cancer.py --mode='pre' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./Prediction/Img_prediction_postive/'  --Pre_savepath='./Prediction/Img_prediction_postive_pre/' --test_path='./dataset/testcancer_pre/'
   
   # --------------------
   python ../main_cancer.py --mode='test' --model_type='R2AttU_Net' --testmodel_path='./models/R2AttU_Net-100-0.0000-78-0.4701_re_pretrained.pkl' --Img_savepath='./Prediction/forwardoutputTest/' --Mask_savepath='./Prediction/forwardoutputTest_GT/' --Pre_savepath='./Prediction/forwardoutput_pre/' --test_path='./dataset/testcancer_all/' --threshold=90
 --
### 生成ISIC数据集

    python ../dataset.py --train_ratio=0.899 --valid_ratio=0.1 --test_ratio=0.001