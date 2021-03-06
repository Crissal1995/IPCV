import json
# Dynamic library search
import importlib
lib_found = importlib.util.find_spec("data_utils")
found = lib_found is not None
if found:
    from data_utils.data_loader import image_segmentation_generator, \
        verify_segmentation_dataset
else:
    from .data_utils.data_loader import image_segmentation_generator, \
        verify_segmentation_dataset
import glob
import six
from keras.callbacks import Callback
import re

from time import time
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        #print(re.sub('\D', '',path))
        #print(path.replace(checkpoints_path, "").strip("."))
        return re.sub('\D', '',path)

    # Get all matching files
    
    all_checkpoint_files = glob.glob(checkpoints_path + "cp.*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    print(all_checkpoint_files)
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path
        
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        if self.checkpoints_path is not None and (epoch+1)%5 == 0:
            self.model.save_weights(self.checkpoints_path + "cp." + str(epoch))
            print("saved ", self.checkpoints_path + "cp." + str(epoch))
            TensorBoard(log_dir='logs/{}'.format(time()))
     
import numpy as np
def prunew(layer_weights,percentage):
    a = len(layer_weights)
    N = a*(100-percentage)/100
    N = int(round(N))
    final_list = []
    l2 = layer_weights.flatten()
    l3 = layer_weights.flatten()
  
    for i in range(0, N):
        max1 = 0
        ind = 0
          
        for j in range(len(l2)):  
            l2[j]
            if abs(l2[j]) > abs(max1):
              
                max1 = l2[j];
                ind = j;
        l2 = np.delete(l2,ind);           
        final_list.append(max1) 
    t = min(map(abs, final_list))
    for k in range(len(l3)):
      if abs(l3[k]) < t:
        l3[k] = 0; 
      else:
        pass
    return l3

def prune(model,percentage=90):
    w = model.get_weights()
    we = []
    wo = []
    wep = []
    for i in range(len(w)):
      if i%2==0:
        we.append(w[i])
      else:
        wo.append(w[i])
    
    for z in we:
      if len(z)>100:
        pr = prunew(z,percentage)
        prr = np.reshape(pr,z.shape)
        wep.append(prr)
      else:
        wep.append(z)
    wprunedfinal = []
    for i in range(5):
      wprunedfinal.append(wep[i])
      wprunedfinal.append(wo[i])
    
    pmodel = model
    pmodel.set_weights(wprunedfinal)
    return pmodel


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          optimizer_name='adadelta',
          loss_type='categorical_crossentropy',
          metrics_used=['accuracy'],
          do_augment=False,
          augmentation_name="aug_all"):

    lib_found = importlib.util.find_spec("models")
    found = lib_found is not None
    if found:
        from models.all_models import model_from_name
    else:
        from .models.all_models import model_from_name
        
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None
        
    # def compile_model(model):
    #      model.compile(loss=loss_type,
    #                   optimizer=optimizer_name,
    #                   metrics=metrics_used)
         
    # def finetune_model(model, initial_epoch, finetune_epochs):
    #     if not validate:
    #         history = model.fit_generator(train_gen, steps_per_epoch,
    #                                       epochs=finetune_epochs, callbacks=callbacks,
    #                                       initial_epoch=initial_epoch)
    #     else:
    #         history = model.fit_generator(train_gen,
    #                             steps_per_epoch,
    #                             validation_data=val_gen,
    #                             validation_steps=val_steps_per_epoch,
    #                             epochs=finetune_epochs, callbacks=callbacks,
    #                             use_multiprocessing=gen_use_multiprocessing,
    #                             initial_epoch=initial_epoch)

    if optimizer_name is not None:

        # if ignore_zero_class:
        #     loss_k = masked_categorical_crossentropy
        # else:
        #     #loss_k = 'categorical_crossentropy'
        #     loss_k = jaccard_distance

        # model.compile(loss=loss_k,
        #               optimizer=optimizer_name,
        #               #metrics=['accuracy'])
        #               metrics=['accuracy', metrics.MeanIoU(name='model_iou', num_classes=n_classes)])
        #compile_model(model)
        model.compile(loss=loss_type, optimizer=optimizer_name, metrics=metrics_used)

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    callbacks = [
        CheckpointsCallback(checkpoints_path),
        CSVLogger(checkpoints_path+model.model_name+'_training.csv')
    ]

    if not validate:
        history = model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing)
        
    # if finetune_epochs > 0:
        
        # pruning = ridurre.KMeansFilterPruning(0.6, compile_model, 
        #                                       finetune_model, finetune_epochs,
        #                                       maximum_pruning_percent=0.4,
        #                                       maximum_prune_iterations=12)
        # model, num = pruning.run_pruning(model)
        # print(model.summary())
    
    # list all data in history
    if epochs > 0:
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['masked_categorical_accuracy'])
        plt.plot(history.history['val_masked_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('accuracy.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('loss.png')
        # summarize history for loss
        plt.plot(history.history['mean_iou'])
        plt.plot(history.history['val_mean_iou'])
        plt.title('model mean IOU')
        plt.ylabel('mean_iou')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('mean_iou.png')
