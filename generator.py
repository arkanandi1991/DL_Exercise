import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path=file_path
        self.label_path=label_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.rotation=rotation
        self.mirroring=mirroring
        self.shuffle=shuffle
        self.batch_next=None
        self.additional_image_numbers=None
        self.keys=None
        self.choice_list=None
        self.choice_random=None
        self.count=0
        self.choice=None
        self.choice_list_angle=None
        self.choice_random_angle=None
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        with open(self.label_path) as file:
            data=json.load(file)
        self.initial_data=[]
        for i in data:
            self.initial_data.append(data[i])
        self.labels=np.array(self.initial_data)
        self.images_path=[self.file_path + name + ".npy" for name in data.keys()]
        self.images=np.array([np.load(j) for j in self.images_path])
        self.arranged_labels=np.arange(len(self.labels))
        self.additional_image_numbers=(len(self.labels) - self.batch_size)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        if self.shuffle is False:
            if self.count==0:
                self.count=self.count+1
            else:
                self.count=self.count+1
                self.images_path=list(np.roll(self.images_path, self.additional_image_numbers))
                self.images=np.array([np.load(k) for k in self.images_path])
                self.labels=np.array(list(np.roll(self.labels, self.additional_image_numbers)))
                self.data=(self.images, self.labels)
            self.batch_next=self.images[0:self.batch_size], self.labels[0:self.batch_size]
        elif self.shuffle is True:
            self.count=self.count + 1
            np.random.shuffle(self.arranged_labels)
            index=self.arranged_labels[0:self.batch_size]
            image=np.array([self.images[i] for i in index])
            label=np.array([self.labels[j] for j in index])
            self.batch_next=image, label

        new_batch_next=list(self.batch_next[0])

        for i, imgs in enumerate(new_batch_next):
            if (list(np.shape(imgs)) != self.image_size):
                resized_img = np.resize(imgs, (self.image_size[0], self.image_size[1],3))
                new_batch_next[i] = resized_img
        self.batch_next = np.array(new_batch_next), self.batch_next[1]

        if self.mirroring is True:
            self.choice = 1
            for i in range(self.batch_size):
                self.choice_list=np.arange(0, 100).tolist()
                self.choice_random=np.random.choice(self.choice_list)
                if (self.choice_random % 2)!=0:
                    self.batch_next[0][i]=self.augment(self.batch_next[0][i],self.choice)
        if self.rotation is True:
            self.choice = 0
            for i in range(self.batch_size):
                self.batch_next[0][i]=self.augment(self.batch_next[0][i],self.choice)
        return self.batch_next

    def augment(self, img, choice_of_action):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        if choice_of_action==1:
            return np.flip(img,axis=1)
        elif choice_of_action==0:
            self.choice_list_angle = np.arange(0,3).tolist()
            self.choice_random_angle = np.random.choice(self.choice_list_angle)
            if self.choice_random_angle==0:#rotate 90 degree
                image1=np.rot90(img)
                return image1
            elif self.choice_random_angle==1:#rotate 180 degree
                image1=np.rot90(img)
                image2=np.rot90(image1)
                return image2
            elif self.choice_random_angle==2:#rotate 270 degree
                image1=np.rot90(img)
                image2=np.rot90(image1)
                image3=np.rot90(image2)
                return image3
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        my_dict=self.class_dict
        return my_dict[x]
    def show(self):
    #     # In order to verify that the generator creates batches as required, this functions calls next to get a
    #     # batch of images and labels and visualizes it.
    #     #TODO: implement show method
        for i in range(self.batch_size):
            nrows=np.math.ceil(np.math.sqrt(self.batch_size))
            ncols=np.math.ceil(np.math.sqrt(self.batch_size))
            index=i+1
            plt.subplot(nrows, ncols, index)
            name_of_class=self.class_name(self.batch_next[1][i])
            plt.title(name_of_class)
            plt.axis('off')
            image_to_show=self.batch_next[0][i]
            plt.imshow(image_to_show)
        plt.show()