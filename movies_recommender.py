import os
from os.path import join 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class Recommender:
    def __init__(self):
        self.R = None
        self.Y = None
        self.n_m = None
        self.n_u = None

    def load_dataset(self):
        current_dir = os.path.abspath(".")
        data_dir = join(current_dir, 'data')
        file_name = join(data_dir,"ex8_movies.mat")
        mat_dict = sio.loadmat(file_name)
        # print("mat_dict.keys() : ",mat_dict.keys())
        self.R = mat_dict["R"] # R[i,j] is 1 if user j has rated movie i , R[i,j] is 0 otherwise
        self.Y = mat_dict["Y"] # rating : 1,2,3,4,5 (0 relate to non rating , same as in R)
        self.compare_no_rating()
        self.n_m = self.Y.shape[0] # number of movies : 1682 -> num rows
        self.n_u = self.Y.shape[1] # number of users : 943 -> num colmns
        # print(f"R.shape : {R.shape} , Y.shape : {Y.shape} ")

    def compare_no_rating(self):
        v1 = np.where(self.Y.reshape(-1) == 0)
        v2 = np.where(self.R.reshape(-1) == 0)
        print(np.array_equal(v1[0], v2[0]))

    def plot_hist(self):
        Y_flat = self.Y.reshape(-1)
        _ , (axs1,axs2) =  plt.subplots(2)

        # first subplot
        axs1.hist(Y_flat)
        axs1.set_title("Y (0,1,2,3,4,5) , 0 means - no rating given")
        axs1.set_xlabel('rating')
        axs1.set_ylabel('count')

        # second subplot
        indices = np.where(Y_flat > 0)
        axs2.set_title("Y (1,2,3,4,5)")
        axs2.set_xlabel('rating')
        axs2.set_ylabel('count')
        axs2.hist(Y_flat[indices])

        plt.tight_layout()
        plt.show()

obj = Recommender()
obj.load_dataset()
obj.plot_hist()