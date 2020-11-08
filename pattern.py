import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self,resolution,tile_size):
        self.tile_size=tile_size
        self.resolution=resolution
        self.output=None
    def draw(self):
        size_of_board=int(self.resolution/self.tile_size)
        num_of_tiles=int(np.floor((size_of_board)/2))
        square_white = np.ones((self.tile_size, self.tile_size))
        square_black=np.zeros((self.tile_size,self.tile_size))
        sub_pattern1=np.append(square_black,square_white,axis=0).T
        sub_pattern2=np.append(square_white,square_black,axis=0).T
        intermediate_pattern1=(np.tile(sub_pattern1,num_of_tiles).T)
        intermediate_pattern2=(np.tile(sub_pattern2,num_of_tiles).T)
        intermediate_pattern=np.append(intermediate_pattern1,intermediate_pattern2,axis=1)
        pattern_final=np.tile(intermediate_pattern,num_of_tiles)
        self.output=np.copy(pattern_final)
        return np.copy(pattern_final)
    def show(self,input):
        plt.imshow(input,cmap="gray")
        plt.show()

class Spectrum():
    def __init__(self,resolution):
        self.resolution=resolution
        self.output=None
    def draw(self):
        range_of_colors=np.linspace(0,1,self.resolution)
        image_canvas=np.zeros((self.resolution,self.resolution,3))
        r=np.tile(range_of_colors,(self.resolution,1))
        g_normal=np.tile(range_of_colors,(self.resolution,1))
        g=g_normal.T
        b_normal=np.tile(range_of_colors,(self.resolution,1))
        b=np.flip(b_normal,axis=1)
        image_canvas[:,:,0]=r
        image_canvas[:,:,1]=g
        image_canvas[:,:,2]=b
        self.output=image_canvas
        return np.copy(image_canvas)
    def show(self):
        plt.imshow(self.output)
        plt.show()

class Circle():
    def __init__(self,resolution,radius,position):
        self.resolution=resolution
        self.center_circle=position
        self.radius=radius

    def draw(self):
        x,y=np.meshgrid(np.arange(self.resolution),np.arange(self.resolution))
        circle_ret =np.zeros((self.resolution,self.resolution))
        dist_from_center =np.sqrt(((x-self.center_circle[0])**2)+((y-self.center_circle[1])**2))
        circle_ret[np.where(dist_from_center<self.radius)]=1
        self.output = circle_ret
        return np.copy(self.output)
    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()