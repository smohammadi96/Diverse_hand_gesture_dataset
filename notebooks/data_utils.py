import os
import pickle
import numpy as np

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(imagedir):
    image_paths = []
    if os.path.isdir(imagedir):
        images = os.listdir(imagedir)
        image_paths = [os.path.join(imagedir,img) for img in images]
    return image_paths

def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        imagedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(imagedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset
	
	
def get_files(path):
    dataset = get_dataset(path)
    labels = []
    train_files = []
    n_of_classes = len(dataset)
    for i in range(n_of_classes):
        labels += [dataset[i].name] * len(dataset[i].image_paths)
        train_files += dataset[i].image_paths
    return train_files, labels
	
def get_labels(base_dir = "./models_new2"):
    lbl_dir = os.path.join(base_dir, "labels")
    with open(lbl_dir, "rb") as f:
        label = pickle.load(f)
    return label
	
def get_feature_files(base_dir = "./models_new2"):
    feature_files = os.listdir(base_dir)
    feature_files.remove("labels")
    return feature_files
	
def get_feature_by_name(feature_file, base_dir = "./models_new2"):
    path = os.path.join(base_dir, feature_file)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data)