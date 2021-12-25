import os
import cv2
import imageio 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# setting the path to our files
import os


PATH = "D:\iti\SelfStudy\Semantic_segmentation\Semantic_Segmentation_Models\CamVid"
labels = pd.read_csv(os.path.join(PATH ,'class_dict.csv'), index_col =0)
id2code={i:tuple(labels.loc[cl, :]) for i,cl in enumerate(labels.index)}

# changing mask
def preprocess_mask(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 720 x 960 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id  {0:(r,g,b)}
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)  #(720,960,32)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )   #(720,960,3)
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def trainGenerator(train_path,image_folder,mask_folder,aug_dict_img,aug_dict_msk,batch_size,image_color_mode = "rgb",
                    mask_color_mode = "rgb",target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
   
    Inputs :
        train_path : String - Root folder path
        image_folder : String - training images folder name
        mask_folder : String - labels folder name
        aug_dict :  dict - dictionary containg augmentations for train images
        aug_dict_msk : dict - dictionary containg augmentations for labels images
        batch_size : int - batch size
        image_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training images
                            default : 'rgb'
        mask_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training labels
                            default : 'rgb'
        target_size : tuple - required image size
                        default : (512,512)
        seed : int - RNG seed to fix image-label pair generation
        
    Returns :
        Generator object                  
        
        
    '''
    image_datagen = ImageDataGenerator(**aug_dict_img)
    mask_datagen = ImageDataGenerator(**aug_dict_msk)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask_img=[preprocess_mask(mask[i]) for i in range(mask.shape[0])]
        yield(img,np.array(mask_img))
        
        
        
        
def validationGenerator(val_path,image_folder,mask_folder,aug_dict_img,aug_dict_msk,batch_size,image_color_mode = "rgb",
                    mask_color_mode = "rgb",target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    
    Inputs :
        val_path : String - Root folder path
        image_folder : String - validation images folder name
        mask_folder : String - labels folder name
        aug_dict :  dict - dictionary containg augmentations for validation images
        aug_dict_msk : dict - dictionary containg augmentations for labels images
        batch_size : int - batch size
        image_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training images
                            default : 'rgb'
        mask_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training labels
                            default : 'rgb'
        target_size : tuple - required image size
                        default : (512,512)
        seed : int - RNG seed to fix image-label pair generation
        
    Returns :
        Generator object   
   
    '''
    image_datagen = ImageDataGenerator(**aug_dict_img)
    mask_datagen = ImageDataGenerator(**aug_dict_msk)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask_img=[preprocess_mask(mask[i]) for i in range(mask.shape[0])]
        yield(img,np.array(mask_img))
        
        
        
def predict_visualize(image_path,model,image_size = (256,256,3),n_classes = 32,alpha = 0.7,plot = False):
    '''
    Predicts image mask and make overlayed mask for visaualization
    inputs :
        image_path : path for image to be predicted
        alpha : alpha value for mask overlay
        
    returns :
        pred_vis : ndarray - predicted RGB mask image
        image : ndarray - input image to the model
        
    '''
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,image_size[:-1],interpolation = cv2.INTER_AREA)
    
    pred = model.predict(np.expand_dims(image,0)/255)
    pred = np.reshape(pred,(image_size[0],image_size[1],n_classes))
    pred = onehot_to_rgb(pred,id2code)
    pred_vis = np.reshape(pred,image_size)

    vis = cv2.addWeighted(image,1.,pred_vis,alpha,0, dtype = cv2.CV_32F)/255
    
    if plot :    
        fig,ax = plt.subplots(1,3)
        fig.set_figwidth(20)
        fig.set_figheight(5)
        ax[0].imshow(image)
        ax[1].imshow(pred_vis)
        ax[2].imshow(vis)
        
        ax[0].title.set_text('Image')
        ax[1].title.set_text('Predicted mask')
        ax[2].title.set_text('masked image')
    
    
    return pred_vis,image
        