# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:39:31 2020

@author: UC
"""


import json
import os
from models.all_models import model_from_name
from train import find_latest_checkpoint
#from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.fcn import fcn_32
from keras_segmentation.models.unet import vgg_unet


# from .data_utils.data_loader import get_image_array, get_segmentation_array,\
#     DATA_LOADER_SEED, class_colors, get_pairs_from_paths
# from .models.config import IMAGE_ORDERING




# def find_latest_checkpoint(checkpoints_path, fail_safe=True):

#     def get_epoch_number_from_path(path):
#         print(re.sub('\D', '',path))
#         #print(path.replace(checkpoints_path, "").strip("."))
#         return re.sub('\D', '',path)

#     # Get all matching files
#     all_checkpoint_files = glob.glob(checkpoints_path + ".*")
#     # Filter out entries where the epoc_number part is pure number
#     all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
#                                        .isdigit(), all_checkpoint_files))
#     print(all_checkpoint_files)
#     if not len(all_checkpoint_files):
#         # The glob list is empty, don't have a checkpoints_path
#         if not fail_safe:
#             raise ValueError("Checkpoint path {0} invalid"
#                              .format(checkpoints_path))
#         else:
#             return None

#     # Find the checkpoint file with the maximum epoch
#     latest_epoch_checkpoint = max(all_checkpoint_files,
#                                   key=lambda f:
#                                   int(get_epoch_number_from_path(f)))
#     return latest_epoch_checkpoint

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.pretrained import resnet_pspnet_VOC12_v0_1
from keras_segmentation.models.pspnet import pspnet_50
from keras_segmentation.models.pspnet import resnet50_pspnet
from keras import backend as K
from keras import metrics
from keras import losses
from keras import optimizers
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

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis = -1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

def jaccard_crossentropy(out, tar):
    return losses.categorical_crossentropy(out, tar) + jaccard_distance(out, tar)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_crossentropy(out, tar):
    return losses.categorical_crossentropy(out, tar) + dice_coef_loss(out, tar)

pretrained_model = pspnet_50_ADE_20K()
#pretrained_model = resnet_pspnet_VOC12_v0_1()
model = pspnet_50( n_classes=38 ) # accuracy: 0.5348 10 epochs
#model = resnet50_pspnet( n_classes=38 ) # accuracy: 0.5154 - frequency_weighted_IU': 0.3473552042914965, 'mean_IU': 0.10207884596666351

transfer_weights( pretrained_model , model  ) # transfer weights from pre-trained model to your model


trainingMode = True
evaluate = False

if trainingMode:
    #model = fcn_32(n_classes=38 ,  input_height=224, input_width=320  )
    # model = vgg_unet(n_classes=38 ,  input_height=416, input_width=608  )
    #model = vgg_unet(n_classes=38 ,  input_height=416, input_width=608  )
    
    opt = optimizers.Adam(learning_rate=0.005)
    if os.name == 'nt':
        init = "C:/Users/UC/Desktop/"
    else:
        init = "/Users/salvatorecapuozzo/Desktop/"
        
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
        val_batch_size = 1,
        n_classes = 38,
        validate = True,
        verify_dataset = False,
        optimizer_name = opt,
        loss_type = dice_crossentropy,
        metrics_used = ['accuracy', metrics.MeanIoU(name='mean_iou', num_classes=n_classes)],
        do_augment = True,
        gen_use_multiprocessing = False,
        ignore_zero_class = False,
        epochs = 1
    )
    
    # print(model.summary())
    # print(model.evaluate_segmentation( 
    #         inp_images_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/rgb/", 
    #         annotations_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/seg/"
    #     ))
else:
    #checkpoints_path = "C:/Users/UC/Desktop/image-segmentation-keras-master/sun_checkpoints_pspnet_4/"
    checkpoints_path = "/Users/salvatorecapuozzo/Desktop/new_jce/"
    #from models.all_models import model_from_name
    #model_from_name["vgg_unet"] = unet.vgg_unet
    full_config_path = checkpoints_path+"_config.json"
    assert (os.path.isfile(full_config_path)
    #assert (os.path.isfile("./"+checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(full_config_path, "r").read())
    #latest_weights = find_latest_checkpoint("./"+checkpoints_path)
    latest_weights = find_latest_checkpoint(checkpoints_path)
    #assert (os.path.isfile("./"+checkpoints_path+".4")
            #), "Weights not found."
    #latest_weights = checkpoints_path+".4"
    assert (latest_weights is not None), "Weights not found."
    assert (os.path.isfile(latest_weights)), "Weights not found."
    #model = model_from_name[model_config['model_class']](
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    print(model.summary())
    
    
    if checkpoints_path == "C:/Users/UC/Desktop/image-segmentation-keras-master/checkpoints/":
        inp_dir = "C:/Users/UC/Desktop/image-segmentation-keras-master/dataset1/images_prepped_test/"
        ann_dir = "C:/Users/UC/Desktop/image-segmentation-keras-master/dataset1/annotations_prepped_test/"
        
        out = model.predict_segmentation(
            inp=inp_dir+"0016E5_07965.png",
            out_fname="C:/Users/UC/Desktop/image-segmentation-keras-master/tmp/out.png"
        )
        
        import matplotlib.pyplot as plt
        plt.imshow(out)
        
        # evaluating the model 
        if evaluate:
            print(model.evaluate_segmentation( inp_images_dir=inp_dir  , annotations_dir=ann_dir ) )
    
    elif checkpoints_path == "/Users/salvatorecapuozzo/Desktop/new_jce/":
        sun_inp_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/rgb/"
        sun_ann_dir = "/Users/salvatorecapuozzo/Desktop/sunrgb/test/seg/"
        
        out = model.predict_segmentation(
            inp=sun_inp_dir+"img_00005.jpg",
            out_fname="/Users/salvatorecapuozzo/Desktop/sun_out_29_05.png"
        )
        
        import matplotlib.pyplot as plt
        plt.imshow(out)
        
        # evaluating the model
        if evaluate:
            print(model.evaluate_segmentation( inp_images_dir=sun_inp_dir  , annotations_dir=sun_ann_dir ) )