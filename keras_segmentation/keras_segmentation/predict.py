import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

# Dynamic library search
import importlib
lib_found = importlib.util.find_spec("data_utils")
found = lib_found is not None
if found:
    from train import find_latest_checkpoint
    from data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
    from models.config import IMAGE_ORDERING
else:
    from .train import find_latest_checkpoint
    from .data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
    from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)


def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile("./"+checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open("./"+checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint("./"+checkpoints_path)
    #assert (os.path.isfile("./"+checkpoints_path+".4")
            #), "Weights not found."
    #latest_weights = checkpoints_path+".4"
    assert (latest_weights is not None), "Weights not found."
    assert (os.path.isfile(latest_weights)), "Weights not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    print(model.summary())
    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def get_bw_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*c).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*c).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*c).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None,
                           grayscale=False):

    if n_classes is None:
        n_classes = np.max(seg_arr)
        
    if grayscale:
        seg_img = get_bw_segmentation_image(seg_arr, n_classes, colors=colors)
    else:
        seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    seg_img_gray = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height,
                                     grayscale=True)
    
    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height,
                                     grayscale=False)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)
        cv2.imwrite(out_fname[:-4]+'_gray.png', seg_img_gray)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)

    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=True,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes

    cap, video, fps = set_video(inp, output)
    while(cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            pr = predict(model=model, inp=frame)
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
                )
        else:
            break
        print("FPS: {}".format(1/(time() - prev_time)))
        if output is not None:
            video.write(fused_img)
        if display:
            cv2.imshow('Frame masked', fused_img)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break
    cap.release()
    if output is not None:
        video.release()
    cv2.destroyAllWindows()

def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None,
             zero_handling="None", split_factor=1):
             #, optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy']):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir, split_factor=split_factor)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes-1)
    fp = np.zeros(model.n_classes-1)
    fn = np.zeros(model.n_classes-1)
    n_pixels = np.zeros(model.n_classes-1)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):
            # Ignoring zero class (ambiguous)
            if cl_i > 0:
                i = cl_i - 1
                tp[i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                n_pixels[i] += np.sum(gt == cl_i)
                zero_total = np.sum(gt == 0)
                zero_positive = np.sum((pr == cl_i) * (gt == 0))
                #zero_negative = np.sum((pr != cl_i) * (gt == 0))
                
                
                # unique, counts = np.unique(gt, return_counts=True)
                # occurs = dict(zip(unique, counts))
                # If zero_handling is none of these, it will be worst case,
                # where zero classes are always classified as wrong
                # if zero_handling == "MeanOnPrevData":
                #     # Zero class has a weight of 1/(num_classes-1)
                #     tp[i] += 0.5*occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     fn[i] += 0.5*occurs[0]/(model.n_classes-1)
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                # elif zero_handling == "WeightedOnPrevData":
                #     # Zero class has a weight that is proportional to
                #     # number of occurences of a certain prediction
                #     tp_score = tp[i] / (tp[i] + fn[i] + 0.000000000001)
                #     fn_score = fn[i] / (tp[i] + fn[i] + 0.000000000001)
                #     tp[i] += tp_score*occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     fn[i] += fn_score*occurs[0]/(model.n_classes-1)
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                # elif zero_handling == "TrustfulOnPrevData":
                #     # Zero class has always weight 1 for every prediction
                #     tp[i] += occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     #fn[i] += 0.5*occurs[0]
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                if zero_handling == "Mean":
                    # Zero class has a weight of 1/(num_classes-1)
                    tp[i] += 0.5*zero_positive
                    fp[i] -= zero_positive
                    fn[i] += 0.5*zero_positive
                    n_pixels[i] += zero_positive
                elif zero_handling == "Weighted":
                    tp_score = zero_positive / (zero_total + 0.000000000001)
                    tp[i] += zero_positive*tp_score
                    fp[i] -= zero_positive
                    fn[i] += zero_positive*(1-tp_score)
                    n_pixels[i] += zero_positive
                elif zero_handling == "Trustful":
                    tp[i] += zero_positive
                    fp[i] -= zero_positive
                    #fn[i] += zero_negative
                    n_pixels[i] += zero_positive
                #elif zero_handling == "None":
                    
                # elif zero_handling == "Idealistic":
                    # conf_score = (zero_positive / (zero_total + 0.000000000001))
                    # tp[i] += occurs[0]*conf_score
                    # fp[i] -= occurs[0]*conf_score
                    # #fn[i] += zero_positive*(conf_score-1)
                    # n_pixels[i] += occurs[0]*conf_score

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    #model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #results = model.evaluate(inp_images_dir, annotations_dir)

    return {
        #"loss": results[0],
        #"accuracy": results[1],
        "frequency_weighted_IoU": frequency_weighted_IU,
        "mean_IoU": mean_IU,
        "class_wise_IoU": cl_wise_score
    }

def evaluate_segmentation(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None,
             zero_handling="None", split_factor=1):
             #, optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy']):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir, split_factor=split_factor)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes-1)
    fp = np.zeros(model.n_classes-1)
    fn = np.zeros(model.n_classes-1)
    n_pixels = np.zeros(model.n_classes-1)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):
            # Ignoring zero class (ambiguous)
            if cl_i > 0:
                i = cl_i - 1
                tp[i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                n_pixels[i] += np.sum(gt == cl_i)
                zero_total = np.sum(gt == 0)
                zero_positive = np.sum((pr == cl_i) * (gt == 0))
                #zero_negative = np.sum((pr != cl_i) * (gt == 0))
                
                
                # unique, counts = np.unique(gt, return_counts=True)
                # occurs = dict(zip(unique, counts))
                # If zero_handling is none of these, it will be worst case,
                # where zero classes are always classified as wrong
                # if zero_handling == "MeanOnPrevData":
                #     # Zero class has a weight of 1/(num_classes-1)
                #     tp[i] += 0.5*occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     fn[i] += 0.5*occurs[0]/(model.n_classes-1)
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                # elif zero_handling == "WeightedOnPrevData":
                #     # Zero class has a weight that is proportional to
                #     # number of occurences of a certain prediction
                #     tp_score = tp[i] / (tp[i] + fn[i] + 0.000000000001)
                #     fn_score = fn[i] / (tp[i] + fn[i] + 0.000000000001)
                #     tp[i] += tp_score*occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     fn[i] += fn_score*occurs[0]/(model.n_classes-1)
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                # elif zero_handling == "TrustfulOnPrevData":
                #     # Zero class has always weight 1 for every prediction
                #     tp[i] += occurs[0]/(model.n_classes-1)
                #     fp[i] -= occurs[0]/(model.n_classes-1)
                #     #fn[i] += 0.5*occurs[0]
                #     n_pixels[i] += occurs[0]/(model.n_classes-1)
                if zero_handling == "Mean":
                    # Zero class has a weight of 1/(num_classes-1)
                    tp[i] += 0.5*zero_positive
                    fp[i] -= zero_positive
                    fn[i] += 0.5*zero_positive
                    n_pixels[i] += zero_positive
                elif zero_handling == "Weighted":
                    tp_score = zero_positive / (zero_total + 0.000000000001)
                    tp[i] += zero_positive*tp_score
                    fp[i] -= zero_positive
                    fn[i] += zero_positive*(1-tp_score)
                    n_pixels[i] += zero_positive
                elif zero_handling == "Trustful":
                    tp[i] += zero_positive
                    fp[i] -= zero_positive
                    #fn[i] += zero_negative
                    n_pixels[i] += zero_positive
                #elif zero_handling == "None":
                    
                # elif zero_handling == "Idealistic":
                    # conf_score = (zero_positive / (zero_total + 0.000000000001))
                    # tp[i] += occurs[0]*conf_score
                    # fp[i] -= occurs[0]*conf_score
                    # #fn[i] += zero_positive*(conf_score-1)
                    # n_pixels[i] += occurs[0]*conf_score

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    #model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #results = model.evaluate(inp_images_dir, annotations_dir)

    return {
        #"loss": results[0],
        #"accuracy": results[1],
        "frequency_weighted_IoU": frequency_weighted_IU,
        "mean_IoU": mean_IU,
        "class_wise_IoU": cl_wise_score,
        "pixels_per_class": n_pixels
    }

def fix_zero_labeling(model=None, seg=None, inp=None, out_fname=None,
            checkpoints_path=None, colors=class_colors):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert (seg is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)
        
    seg = cv2.imread(seg)

    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    
    seg_img_gray = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=False,
                                     show_legends=False,
                                     class_names=None,
                                     prediction_width=None,
                                     prediction_height=None,
                                     grayscale=True)

    
    seg_img_gray = seg*(seg>0)+seg_img_gray*(seg==0)
    
    # seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
    #                                  colors=colors, overlay_img=overlay_img,
    #                                  show_legends=show_legends,
    #                                  class_names=class_names,
    #                                  prediction_width=prediction_width,
    #                                  prediction_height=prediction_height,
    #                                  grayscale=False)

    if out_fname is not None:
        #cv2.imwrite(out_fname, seg_img)
        cv2.imwrite(out_fname, seg_img_gray)


def fix_dataset(model,
                checkpoints_path=None,
                source_path="C:/Users/UC/Desktop/half_sunrgb/", 
                destination_path="C:/Users/UC/Desktop/fixed_segs/"):
    folder_path = source_path+"train/"
    folder = os.listdir(folder_path+"seg/")
    for file in folder:
        fix_zero_labeling(model=model, seg=folder_path+"seg/"+file, inp=folder_path+"rgb/"+file, out_fname=destination_path+"train/"+file,
            checkpoints_path=checkpoints_path)
    
    print("Training set converted")
    
    folder_path = source_path+"val/"
    folder = os.listdir(folder_path+"seg/")
    for file in folder:
        fix_zero_labeling(model=model, seg=folder_path+"seg/"+file, inp=folder_path+"rgb/"+file, out_fname=destination_path+"val/"+file,
            checkpoints_path=checkpoints_path)
        
    print("Validation set converted")
    
    folder_path = source_path+"test/"
    folder = os.listdir(folder_path+"seg/")
    for file in folder:
        fix_zero_labeling(model=model, seg=folder_path+"seg/"+file, inp=folder_path+"rgb/"+file, out_fname=destination_path+"test/"+file,
            checkpoints_path=checkpoints_path)
        
    print("Test set converted")

def display(display_list, image_name, labels, height=8, dest_dir="C:/Users/UC/Desktop/display_segs/"):
  l = len(display_list)
  assert l < 4, 'Fornire in ingresso al più 3 immagini'

  fig, ax = plt.subplots(1, l, figsize=(height*l, height))
  #afig, aax = plt.subplots(1, 1, figsize=(20, 1))
  titles = ['Input Image', 'True Mask', 'Predicted Mask']

  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
  # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.colors.ListedColormap.html
  # ListedColormap per enumerare i colori desiderati
  cmap = mpl.cm.gnuplot
  bounds = np.linspace(0, len(labels), len(labels)+1)
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  labels_dict = dict(zip(range(len(labels)), labels))
  def get_labels(values):
    return [label for key, label in labels_dict.items() if key in values]

  if l == 3 and image_name != None:
      fig.suptitle(image_name,fontsize=16)

  for i in range(l):
    x = display_list[i]
    # il primo controllo non ci fa crashare se abbiamo un'imm a 0 canali (1 implicito)
    # il secondo serve a capire quanti canali ci sono nella terza dim
    rgb = len(x.shape) > 2 and x.shape[2] > 1

    if i == 0:
      x = tf.keras.preprocessing.image.array_to_img(x)
    else:
        if rgb:
          x = x[:,:,0]
            
        x = x.squeeze()
        x = x*255

    title = titles[i]
    ax[i].set_title(title)
    ax[i].imshow(x, cmap=cmap, vmin=0, vmax=len(labels) - 1)
    ax[i].axis('off')
    if i != 0:
      uniques = [e for e in np.unique(x) if np.round(e) == e]
      print('IMAGE: ', title)
      print('LABELS:', get_labels(uniques))
      print()

  ax1 = fig.add_subplot(4,1,4)
  cb = mpl.colorbar.ColorbarBase(
    ax1, cmap=cmap,
    norm=norm,
    boundaries=bounds,
    ticks=bounds,
    spacing='proportional',
    orientation='horizontal'
  )
  
  cb.ax.set_xticklabels(labels, rotation=75, size=8, ha='left')
  
  plt.show()
  fig.set_size_inches(14,10)
  plt.savefig(dest_dir+image_name)