import os
import pdb

import numpy as np
import tensorflow as tf

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    labels_2 = []
    for line in f:
        label_2 = line.strip("\n").split(' ')[-4:]  # label for color and type
        image, mask = line.strip("\n").split(' ')[:-4] # path of image and mask
        images.append(data_dir + image)
        masks.append(data_dir + mask)
        labels_2.append([int(num) for num in label_2])
    return images, masks, labels_2

def read_images_from_disk(input_queue, input_size, random_scale): 
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    label_2 = input_queue[2]
    
    #img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.image.decode_png(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)
    if input_size is not None:
        h, w = input_size
        '''
        if random_scale:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
            w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
            img = tf.image.resize_images(img, new_shape)
            # label must have form of [batch, h, w, c]
            label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
            label = tf.squeeze(label, squeeze_dims=[0]) 
        '''
        # resize_image_with_crop_or_pad accepts 3D-tensor.
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)
        label = tf.image.resize_image_with_crop_or_pad(label, h, w)
    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=[1,1,1], axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    return img, label, label_2

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, random_scale, coord, epoch):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        self.image_list, self.label_list, self.labels_2_array = read_labeled_image_list(self.data_dir, self.data_list)
        self.dataset_size = len(self.image_list)
        #print(self.image_list, self.label_list, self.labels_2_array)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.labels_2 = tf.convert_to_tensor(self.labels_2_array)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.labels_2],
                                                   shuffle=input_size is not None, num_epochs=epoch) # Not shuffling if it is val.
        self.image, self.label, self.label_2 = read_images_from_disk(self.queue, self.input_size, random_scale)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''

        image_batch, label_batch1, label_batch2 = tf.train.batch([self.image, self.label, self.label_2], num_elements)
        return image_batch, label_batch1, label_batch2



# test 
if __name__ == '__main__':

    ## load image_batch and label_batch
    data_dir = '/home/BackUp/docker-file/wjt/Changshu'
    data_list = '/home/BackUp/docker-file/wjt/Changshu/list/train_gt.txt'
    input_size = (512, 640)
    RANDOM_SCALE = 0
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    epoch = 20
    batch_size = 4
    
    reader = ImageReader(
        data_dir,
        data_list,
        input_size,
        RANDOM_SCALE,
        coord,
        epoch)
    image_batch, label_batch1, label_batch2 = reader.dequeue(num_elements = batch_size)

 
    input_queue = [ '/home/BackUp/docker-file/wjt/Changshu/TSD-Lane-00051/TSD-Lane-00051-00000.png', '/home/BackUp/docker-file/wjt/Changshu/label/TSD-Lane-00051/TSD-Lane-00051-00000.png', [1,1,1,1] ]
    img, label1, label2 = read_images_from_disk(input_queue, input_size, False)    
    sess = tf.Session()
    pdb.set_trace()
    #images = sess.run(image_batch)
    #print(images.shape)
    
