# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:39:31 2020

@author: UC
"""


import json
import os
import glob
from models.all_models import model_from_name
from train import find_latest_checkpoint
#from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.fcn import fcn_32
from keras_segmentation.models.unet import vgg_unet

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.pretrained import resnet_pspnet_VOC12_v0_1
from keras_segmentation.models.pspnet import pspnet_50, pspnet_50_sunrgb
from keras_segmentation.models.pspnet import resnet50_pspnet
from keras_segmentation.models._pspnet_2 import Interp
from keras_segmentation.train import prune
from keras_segmentation.predict import display, fix_zero_labeling, fix_dataset, evaluate_segmentation
from keras_segmentation.custom_losses_metrics import jaccard_crossentropy, masked_jaccard_crossentropy, masked_categorical_accuracy, mild_categorical_crossentropy
from keras.layers import Lambda, Input
from keras.utils import get_file
from keras_segmentation.models.segnet import mobilenet_segnet
from keras import backend as K
from keras import metrics
from keras import losses
from keras import optimizers
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels,delete_layer
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
        verify_segmentation_dataset

def pspnet_50_slim(n_classes,  input_height=256, input_width=256):
    from keras_segmentation.models._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "pspnet_50_slim"
    return model

# def jaccard_distance(y_true, y_pred, smooth=100):
#     intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
#     sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis = -1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return (1 - jac) * smooth

# def masked_categorical_crossentropy(gt, pr):
#     from keras.losses import categorical_crossentropy
#     mask = 1 - gt[:, :, 0]
#     return categorical_crossentropy(gt, pr) * mask




#pretrained_model = pspnet_50_ADE_20K()
#pretrained_model = resnet_pspnet_VOC12_v0_1()
#model = pspnet_50_slim( n_classes=38 ) # accuracy: 0.5348 10 epochs
#model = resnet50_pspnet( n_classes=38 ) # accuracy: 0.5154 - frequency_weighted_IU': 0.3473552042914965, 'mean_IU': 0.10207884596666351

#transfer_weights( pretrained_model , model  ) # transfer weights from pre-trained model to your model



    
# def sunrgbize(x):
#     sunrgb_class_range = [1,4,11,8,20,24,16,15,9,63,23,46,87,34,25,19,45,58,28,29,93,6,68,51,90,131,82,146,42,44,13,45,66,48,37,38,116]
#     total_range = range(0,151)
#     sub_range = [item for item in total_range if item not in sunrgb_class_range]

#     concat = x[:,:,:,127,None] #Simulates the empty class
#     for i in sunrgb_class_range:
#         new_slice = x[:,:,:,i-1,None]
#         concat = K.concatenate([concat,new_slice], axis=-1)
    
#     return concat

# def convertToSunRgb(model,distructive=False):
#     sunrgb_class_range = [1,4,11,8,20,24,16,15,9,63,23,46,87,34,25,19,45,58,28,29,93,6,68,51,90,131,82,146,42,44,13,45,66,48,37,38,116]
#     total_range = range(0,151)
#     sub_range = [item for item in total_range if item not in sunrgb_class_range]
#     # for layer in model.layers:
#     #     layer.trainable = False
    
#     # model.trainable = False

#     #inputs = Input(shape=(inp.shape[1],inp.shape[2],inp.shape[3]))
#     #x = model(inputs)
    
    
#     inp = model.input
    
#     model.layers.pop()
#     model.layers.pop()
#     model.layers.pop()

    
#     x = model.layers[-1].output
#     print(x)
    
#     if distructive:
#         # Not working
#         new_model = delete_channels(model,x,sub_range)
#     else:
#         x = Lambda(sunrgbize, name='class_filter')(x)
        
#         x = Interp([473, 473])(x)
        
#         new_model = get_segmentation_model(inp,x)
    
#     return new_model

def pspnet_50_ADE_20K_SUNRGB(height=473,width=473):

    model_url = "https://www.dropbox.com/s/" \
                "0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1"
    latest_weights = get_file("pspnet50_ade20k.h5", model_url)
    
    model = pspnet_50_sunrgb(input_height=height,input_width=width)
    model.load_weights(latest_weights)
    return model


mode = 1
trainingFromInit = True
evaluate = True
writeOnNotes = True

if os.name == 'nt':
    init = "C:/Users/UC/Desktop/"
else:
    init = "/Users/salvatorecapuozzo/Desktop/"

if mode == 0:
    #model = fcn_32(n_classes=38 ,  input_height=224, input_width=320  )
    # model = vgg_unet(n_classes=38 ,  input_height=416, input_width=608  )
    #model = vgg_unet(n_classes=38 ,  input_height=416, input_width=608  )
    
    if trainingFromInit:
        #model = pspnet_50_slim( n_classes=38 ) # accuracy: 0.5348 10 epochs
        #model = pspnet_50_ADE_20K_SUNRGB()
        #model = convertToSunRgb(model)
        #print(model.summary())
        #prun_schedule = PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5,begin_step=2000,end_step=4000)
        #model = prune_low_magnitude(model, pruning_schedule=prun_schedule)
        model = pspnet_50( n_classes=38 )
        pretrained_model = pspnet_50_ADE_20K()
        
        tf.keras.utils.plot_model(
            model,
            to_file=init+"pspnet_50.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )
        tf.keras.utils.plot_model(
            pretrained_model,
            to_file=init+"pspnet_50_ade_20k.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )
        transfer_weights( pretrained_model, model, trainable_source=True  )
        
        opt = optimizers.Adam(learning_rate=0.00001)
    else:
        checkpoints_path = init+"pspnet_50_slim_checkpoint_50/"
        full_config_path = checkpoints_path+"_config.json"
        assert (os.path.isfile(full_config_path)
        #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
                ), "Checkpoint not found."
        model_config = json.loads(
            open(full_config_path, "r").read())
        latest_weights = find_latest_checkpoint(checkpoints_path)
        assert (latest_weights is not None), "Weights not found."
        assert (os.path.isfile(latest_weights)), "Weights not found."
        #model = model_from_name[model_config['model_class']](
        
        model = model_from_name[model_config['model_class']](
            model_config['n_classes'], input_height=model_config['input_height'],
            input_width=model_config['input_width'])
        print("loaded weights ", latest_weights)
        model.load_weights(latest_weights)
        opt = optimizers.Adam(learning_rate=0.00001)
    
        
    train_images =  init+"sunrgb/train/rgb/"
    train_annotations = init+"sunrgb/train/seg/"
    val_images =  init+"sunrgb/val/rgb/",
    val_annotations = init+"sunrgb/val/seg/",
    train_images_2 =  init+"half_sunrgb/train/rgb/"
    train_annotations_2 = init+"half_sunrgb/train/seg/"
    val_images_2 =  init+"half_sunrgb/val/rgb/",
    val_annotations_2 = init+"half_sunrgb/val/seg/",
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    model.train(
        train_images =  train_images,
        train_annotations = train_annotations,
        val_images = init+"sunrgb/val/rgb/",
        val_annotations = init+"sunrgb/val/seg/",
        checkpoints_path = init,
        batch_size = 2,
        steps_per_epoch = 512,
        val_batch_size = 2,
        n_classes = 38,
        validate = True,
        verify_dataset = False,
        optimizer_name = opt,
        loss_type = mild_categorical_crossentropy,
        metrics_used = [masked_categorical_accuracy, metrics.MeanIoU(name='mean_iou', num_classes=n_classes)],
        do_augment = False,
        gen_use_multiprocessing = False,
        epochs = 5
    )
    
    # print(model.summary())
    # print(model.evaluate_segmentation( 
    #         inp_images_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/rgb/", 
    #         annotations_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/seg/"
    #     ))
elif mode == 1:
    import sys
    if writeOnNotes:
        f = open("C:/Users/UC/Desktop/test_models.out", 'w')
        sys.stdout = f
    
    sun_inp_dir = "C:/Users/UC/Desktop/sunrgb/test/rgb/"
    sun_ann_dir = "C:/Users/UC/Desktop/sunrgb/test/seg/"
    labels = json.load(open('labels.json'))
        
    split_factor = 1
    #chosen_imgs = ["img_00005.png","img_00027.png","img_02592.png","img_04521.png","img_06600.png"]
    #chosen_imgs = ["img_00027.png"]
    #chosen_imgs = ["img_07033.png"]
    chosen_imgs = ["img_00005.png"]
        
    checkpoints_path = init+"ipcv_checkpoints/"
    all_checkpoint_folders = glob.glob(checkpoints_path + "*")
    
    from math import ceil
    
    #for j in range(0,5):
    for j in range(1,2):
        dataset_status = "Original"
        if j < 4:
            sun_ann_dir = "C:/Users/UC/Desktop/sunrgb/test/seg/"
            dataset_status = "Original"
        else:
            sun_ann_dir = "C:/Users/UC/Desktop/fixed_segs/test/"
            dataset_status = "Fixed"
        
        if j == 0 or j == 4:
            handling = "None"
        elif j == 1:
            handling = "Mean"
        elif j == 2:
            handling = "Weighted"
        elif j == 3:
            handling = "Trustful"
            
        model_untrained = pspnet_50_ADE_20K_SUNRGB()
        n_classes = 38
        folder_untrained = "pspnet_50_ade20K_untrained"
        
        # evaluating the model 
        if evaluate:
            evaluation_untrained = evaluate_segmentation(model_untrained, inp_images_dir=sun_inp_dir, 
                annotations_dir=sun_ann_dir, zero_handling=handling, split_factor=split_factor)
            print(evaluation_untrained)
            evals = [evaluation_untrained]


        for chosen_img in chosen_imgs: 
            if j == 0 or j == 4 or j == 1:
                i = 3
                size = int(ceil(ceil(len(all_checkpoint_folders)+3)/2))
                fig, axes = plt.subplots(2,size)
                # Immagine Originale
                img = plt.imread(sun_inp_dir+chosen_img)
                axes[0][0].set_title('Original', fontsize=6)
                axes[0][0].imshow(img)
                # Immagine Segmentata
                img2 = plt.imread(sun_ann_dir+chosen_img)
                axes[0][1].set_title('Segmentation', fontsize=6)
                axes[0][1].imshow(img2)
                # Immagine
                print("\nModel Evaluation: "+folder_untrained+"\n")
                out = model_untrained.predict_segmentation(
                    inp=sun_inp_dir+chosen_img,
                    out_fname=init+"/ipcv_out/"+folder_untrained+"_"+dataset_status+"_"+chosen_img
                )
                
                if evaluate:
                    axes[0][2].set_title(
                    folder_untrained+
                    #"\n("+str(round(100*evaluation['loss'])/100)+
                    #" - "+str(round(100*evaluation['accuracy'])/100)+
                    "\n("+str(round(100*evaluation_untrained['frequency_weighted_IoU'])/100)+
                    " - "+str(round(100*evaluation_untrained['mean_IoU'])/100)+")", fontsize=6)
                    axes[0][2].imshow(out)
                else:
                    axes[0][2].set_title(folder_untrained, fontsize=6)
                    axes[0][2].imshow(out)
                
                for folder in all_checkpoint_folders:
                    #from models.all_models import model_from_name
                    #model_from_name["vgg_unet"] = unet.vgg_unet
                    full_config_path = folder+"\_config.json"
                    #print(full_config_path)
                    assert (os.path.isfile(full_config_path)
                    #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
                            ), "Checkpoint not found."
                    model_config = json.loads(
                        open(full_config_path, "r").read())
                    #latest_weights = find_latest_checkpoint("./"+checkpoints_path)
                    latest_weights = find_latest_checkpoint(folder+"/")
                    #assert (os.path.isfile("./"+checkpoints_path+".4")
                            #), "Weights not found."
                    #latest_weights = checkpoints_path+".4"
                    assert (latest_weights is not None), "Weights not found."
                    assert (os.path.isfile(latest_weights)), "Weights not found."
                    #model = model_from_name[model_config['model_class']](
                    
                    if model_config['model_class'] == "" or model_config['model_class'] == "pspnet_50_sunrgb":
                        model = pspnet_50_ADE_20K_SUNRGB()
                        n_classes = 38
                        #model = convertToSunRgb(model)
                    else:
                        model = model_from_name[model_config['model_class']](
                            model_config['n_classes'], input_height=model_config['input_height'],
                            input_width=model_config['input_width'])
                        n_classes = model_config['n_classes']
                    print("loaded weights ", latest_weights)
                    model.load_weights(latest_weights)
                    #print(model.summary())
                    folder = os.path.basename(folder)
                    
                    # evaluating the model 
                    if evaluate:
                        evaluation = evaluate_segmentation(model, inp_images_dir=sun_inp_dir, 
                            annotations_dir=sun_ann_dir, zero_handling="None", split_factor=split_factor)
                        print(evaluation)
                        evals.append(evaluation)
                    
                        
                    print("\nModel Evaluation: "+folder+"\n")
                    out = model.predict_segmentation(
                        inp=sun_inp_dir+chosen_img,
                        out_fname=init+"/ipcv_out/"+folder+"_"+dataset_status+"_"+latest_weights[-2:]+"_"+chosen_img
                    )
                     
                    if evaluate:
                        axes[int(i/size)][int((i)%size)].set_title(
                            folder+"_"+
                            latest_weights[-2:]+
                            #"\n("+str(round(100*evaluation['loss'])/100)+
                            #" - "+str(round(100*evaluation['accuracy'])/100)+
                            "\n("+str(round(100*evaluation['frequency_weighted_IoU'])/100)+
                            " - "+str(round(100*evaluation['mean_IoU'])/100)+")", fontsize=6)
                        axes[int(i/size)][int((i)%size)].imshow(out)
                    else:
                        axes[int(i/size)][int((i)%size)].set_title(folder+"_"+latest_weights[-2:], fontsize=6)
                        axes[int(i/size)][int((i)%size)].imshow(out)
            
                    i += 1
                    
                plt.show()
                fig.set_size_inches(14,10)
                plt.savefig(init+"/ipcv_out/all_models_"+dataset_status+"_"+chosen_img)
            
                if evaluate == False:
                    print("\nImage: "+chosen_img+" - Model: "+folder_untrained+" - Zero handling: "+handling+" - Fixed dataset: "+str(j==5)+"\n")
                    display_list = [plt.imread(sun_inp_dir+chosen_img),
                                plt.imread(sun_ann_dir+chosen_img),
                                plt.imread(init+"/ipcv_out/"+folder_untrained+"_"+dataset_status+"_"+chosen_img[:-4]+"_gray.png")]
                    display(display_list, 
                            folder_untrained+"_"+dataset_status+"_"+chosen_img, labels, height=8)
                
                for folder in all_checkpoint_folders:
                    full_config_path = folder+"\_config.json"
                    #print(full_config_path)
                    assert (os.path.isfile(full_config_path)
                    #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
                            ), "Checkpoint not found."
                    model_config = json.loads(
                        open(full_config_path, "r").read())
                    #latest_weights = find_latest_checkpoint("./"+checkpoints_path)
                    latest_weights = find_latest_checkpoint(folder+"/")
                    #assert (os.path.isfile("./"+checkpoints_path+".4")
                            #), "Weights not found."
                    #latest_weights = checkpoints_path+".4"
                    assert (latest_weights is not None), "Weights not found."
                    assert (os.path.isfile(latest_weights)), "Weights not found."
                    folder = os.path.basename(folder)
                    
                    if evaluate == False:
                        print("\nImage: "+chosen_img+" - Model: "+folder_untrained+" - Zero handling: "+handling+" - Fixed dataset: "+str(j==5)+"\n")
                        display_list = [plt.imread(sun_inp_dir+chosen_img),
                                plt.imread(sun_ann_dir+chosen_img),
                                plt.imread(init+"/ipcv_out/"+folder+"_"+dataset_status+"_"+latest_weights[-2:]+"_"+chosen_img[:-4]+"_gray.png")]
                        display(display_list, 
                                folder+"_"+dataset_status+"_"+latest_weights[-2:]+"_"+chosen_img, labels, 
                                height=8)

                #opt = optimizers.Adam(learning_rate=10e-6)
                #loss_type = masked_jaccard_crossentropy
                #metrics_used = [masked_categorical_accuracy, metrics.MeanIoU(name='mean_iou', num_classes=n_classes)]
            else:
                # evaluating the model 
                if evaluate:
                    evaluation = evaluate_segmentation(model, inp_images_dir=sun_inp_dir, 
                        annotations_dir=sun_ann_dir, zero_handling="None", split_factor=split_factor)
                    print(evaluation)
                    evals.append(evaluation)
        j += 1
       
    if writeOnNotes:
        f.close()
        
elif mode == 2:
    #pretrained_model = pspnet_50_ADE_20K()
    #model = pspnet_50( n_classes=38 ) # accuracy: 0.5348 10 epochs
    
    #transfer_weights( pretrained_model , model  ) # transfer weights from pre-trained model to your model
    
    model = pspnet_50_ADE_20K_SUNRGB()
    #print(model.summary())
    #model = convertToSunRgb(model)
    #print(model.summary())

    sun_inp_dir = init+"half_sunrgb/test/rgb/"
    sun_ann_dir = init+"half_sunrgb/test/seg/"
        
    chosen_img = "img_06600.png"
        
    #checkpoints_path = init+"ipcv_checkpoints/"
    #all_checkpoint_folders = glob.glob(checkpoints_path + "*")
    
    
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3)
    img = plt.imread(sun_inp_dir+chosen_img)
    axes[0].set_title('Original', fontsize=6)
    axes[0].imshow(img)
    img2 = plt.imread(sun_ann_dir+chosen_img)
    axes[1].set_title('Segmentation', fontsize=6)
    axes[1].imshow(img2)
    
    out = model.predict_segmentation(
        inp=sun_inp_dir+chosen_img,
        out_fname=init+"/ipcv_out/pspnet_50_ade20k_"+chosen_img
    )
    
    # evaluating the model 
    if evaluate:
        evaluation = model.evaluate_segmentation( inp_images_dir=sun_inp_dir  , annotations_dir=sun_ann_dir ) 
        print(evaluation)
        axes[2].set_title(
            "pspnet_50_ade20k"+
            "_("+str(round(100*evaluation['frequency_weighted_IoU'])/100)+
            " - "+str(round(100*evaluation['mean_IoU'])/100)+")", fontsize=6)
        axes[2].imshow(out)
    else:
        axes[2].set_title("pspnet_50_ade20k", fontsize=6)
        axes[2].imshow(out)
    
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    plt.show()
    fig.set_size_inches(14,10)
    plt.savefig(init+"/ipcv_out/ade20k.png")
    
elif mode == 3:
    print_models = False
    model = pspnet_50_slim(n_classes = 38)
    #print(model.summary())
    
    if print_models:
        tf.keras.utils.plot_model(
            model,
            to_file=init+"pspnet_50_slim.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )
    
    #model1 = pspnet_50_ADE_20K_SUNRGB(height=256,width=256)
    #model1 = convertToSunRgb(model1,distructive=False)
    #print(model1.summary())
    
    sun_inp_dir = init+"sunrgb/test/rgb/"
    sun_ann_dir = init+"sunrgb/test/seg/"
    
    model1 = pspnet_50_ADE_20K_SUNRGB()
    
    fix_dataset(model1,source_path="C:/Users/UC/Desktop/sunrgb/", 
                destination_path="C:/Users/UC/Desktop/fixed_segs/")
    #model2 = convertToSunRgb(model2,distructive=False)
    
    checkpoints_path = init+"ipcv_checkpoints/pspnet_50_20K_fixed_dataset/"
    full_config_path = checkpoints_path+"\_config.json"
    #print(full_config_path)
    assert (os.path.isfile(full_config_path)
    #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(full_config_path, "r").read())
    #latest_weights = find_latest_checkpoint("./"+checkpoints_path)
    latest_weights = find_latest_checkpoint(checkpoints_path+"/")
    #assert (os.path.isfile("./"+checkpoints_path+".4")
            #), "Weights not found."
    #latest_weights = checkpoints_path+".4"
    assert (latest_weights is not None), "Weights not found."
    assert (os.path.isfile(latest_weights)), "Weights not found."
    #model = model_from_name[model_config['model_class']](
    
    ev1 = evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
                   annotations_dir=sun_ann_dir, zero_handling="None", split_factor=4)
    fw1 = ev1['frequency_weighted_IoU']
    print(ev1)

    print("loaded weights ", latest_weights)
    model1.load_weights(latest_weights)
    
    
    
    ev2 = evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
                   annotations_dir=sun_ann_dir, zero_handling="None", split_factor=4)
    fw2 = ev2['frequency_weighted_IoU']
    print(ev2)
    
    # print(evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
    #                annotations_dir=sun_ann_dir, zero_handling="Mean", split_factor=4))
    
    # print(evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
    #                annotations_dir=sun_ann_dir, zero_handling="Weighted", split_factor=4))
    
    # ev3 = evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
    #                annotations_dir=sun_ann_dir, zero_handling="Trustful", split_factor=4)
    # fw3 = ev3['frequency_weighted_IoU']
    # print(ev3)
    # print(fw2*fw3/fw1)
    
    sun_ann_dir="C:/Users/UC/Desktop/fixed_segs/test/"
    
    # Da attivare solo quando si ha solo il dataset con classi anche unlabeled
    # fix_dataset(model1,checkpoints_path)
    
    print(evaluate_segmentation(model1, inp_images_dir=sun_inp_dir, 
                   annotations_dir=sun_ann_dir, zero_handling="None", split_factor=4))
    

    
    if print_models:
        tf.keras.utils.plot_model(
            model1,
            to_file=init+"pspnet_50_ade_20k_resized.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )
    
    # model2 = pspnet_50_ADE_20K_SUNRGB(height=256,width=256)
    # #model2 = convertToSunRgb(model2,distructive=False)
    
    # checkpoints_path = init+"ipcv_checkpoints/pspnet_50_20K_trained/"
    # full_config_path = checkpoints_path+"\_config.json"
    # #print(full_config_path)
    # assert (os.path.isfile(full_config_path)
    # #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
    #         ), "Checkpoint not found."
    # model_config = json.loads(
    #     open(full_config_path, "r").read())
    # #latest_weights = find_latest_checkpoint("./"+checkpoints_path)
    # latest_weights = find_latest_checkpoint(checkpoints_path+"/")
    # #assert (os.path.isfile("./"+checkpoints_path+".4")
    #         #), "Weights not found."
    # #latest_weights = checkpoints_path+".4"
    # assert (latest_weights is not None), "Weights not found."
    # assert (os.path.isfile(latest_weights)), "Weights not found."
    # #model = model_from_name[model_config['model_class']](

    # print("loaded weights ", latest_weights)
    # model2.load_weights(latest_weights)
    
    # evaluation2 = model2.evaluate_segmentation( inp_images_dir=sun_inp_dir  , annotations_dir=sun_ann_dir ) 
    # print(evaluation2)
    
    # opt = optimizers.Adam(learning_rate=0.00001)
    # train_images =  init+"sunrgb/train/rgb/"
    # train_annotations = init+"sunrgb/train/seg/"
    # val_images =  init+"sunrgb/val/rgb/",
    # val_annotations = init+"sunrgb/val/seg/",
    # train_images_2 =  init+"half_sunrgb/train/rgb/"
    # train_annotations_2 = init+"half_sunrgb/train/seg/"
    # val_images_2 =  init+"half_sunrgb/val/rgb/",
    # val_annotations_2 = init+"half_sunrgb/val/seg/",
    # n_classes = model.n_classes
    # input_height = model.input_height
    # input_width = model.input_width
    # output_height = model.output_height
    # output_width = model.output_width

    # model = prune(model2)
    #print(model.summary())
    # model2.train(
    #     train_images =  train_images,
    #     train_annotations = train_annotations,
    #     val_images = init+"sunrgb/val/rgb/",
    #     val_annotations = init+"sunrgb/val/seg/",
    #     checkpoints_path = init,
    #     batch_size = 2,
    #     steps_per_epoch = 512,
    #     val_batch_size = 2,
    #     n_classes = 38,
    #     validate = True,
    #     verify_dataset = False,
    #     optimizer_name = opt,
    #     loss_type = jaccard_crossentropy,
    #     metrics_used = ['accuracy', metrics.MeanIoU(name='mean_iou', num_classes=n_classes)],
    #     do_augment = False,
    #     gen_use_multiprocessing = False,
    #     ignore_zero_class = False,
    #     epochs = 0
    # )
    
    
    # layers1 = model1.layers
    # layers2 = model2.layers
    
    # equal_count = 0
    # total = 0

    # bar = zip(layers1, layers2)
    
    # for l, ll in bar:
    #     for w, ww in zip(list(l.weights),list(ll.weights)):
    #         if not any([w.shape != ww.shape]):
    #             for e, ee in zip(w.numpy().flatten(),ww.numpy().flatten()):
    #                 total += 1
    #                 if abs(e-ee) < 10e-9:
    #                     equal_count += 1
            
    
        
    
    # print(equal_count)
    # print(total)
    #sun_inp_dir = init+"half_sunrgb/test/rgb/"
    #sun_ann_dir = init+"half_sunrgb/test/seg/"
    #evaluation = model.evaluate_segmentation( inp_images_dir=sun_inp_dir  , annotations_dir=sun_ann_dir ) 
    #print(evaluation)
    
elif mode == 4:
    sun_inp_dir = "C:/Users/UC/Desktop/test/rgb/"
    sun_ann_dir = "C:/Users/UC/Desktop/test/seg/"
    pred = "C:/Users/UC/Desktop/ipcv_out/pspnet_50_ade20K_untrained_img_00027_gray.png"
        
    chosen_imgs = ["img_00005.png","img_00015.png","img_00019.png",
                   "img_00022.png","img_00023.png","img_00025.png",
                   "img_00027.png","img_00052.png","img_00055.png",
                   "img_00061.png"]
    
    fig, axes = plt.subplots(2,5)
    i = 0
    for chosen_img in chosen_imgs:
        img = plt.imread(sun_ann_dir+chosen_img)
        axes[int(i/5)][int(i%5)].imshow(img)
        i += 1
        
    plt.show()
    fig.set_size_inches(14,10)
    plt.savefig(init+"/ipcv_out/segmentations.png")
    
elif mode == 5:
    train_list = ["pspnet_50_ade20K_untrained","pspnet_50_89","pspnet_50_20K_trained_1_24",
                  "pspnet_50_20K_trained_2_24","pspnet_50_20K_trained_3_19","pspnet_50_20K_trained_4_24",
                  "pspnet_50_slim_22","pspnet_50_slim_last_24","pspnet_50_slim_new_32"]
    sun_inp_dir = "C:/Users/UC/Desktop/test/rgb/"
    sun_ann_dir = "C:/Users/UC/Desktop/test/seg/"
    pred_dir = "C:/Users/UC/Desktop/ipcv_out/"
    labels = json.load(open('labels.json'))
    for model in train_list:
        display_list = [plt.imread(sun_inp_dir+"img_00027.png"),plt.imread(sun_ann_dir+"img_00027.png"),plt.imread(pred_dir+model+"_img_00027_gray.png")]
        display(display_list, "img_00027.png", labels, height=8, name=model)