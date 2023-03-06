import cv2 
import numpy as np 
import os 
import pickle

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
    
def build_vocabulary(sift, images, vocab_size) -> None:
    # extract features
    features = []

    for img in images:
        img8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, descriptors = sift.detectAndCompute(img8bit, None)
        try:
            if descriptors == None:
                continue 
        except:
            features.append(descriptors)

    all_features = np.vstack(features)
    kmeans = MiniBatchKMeans(n_clusters=vocab_size)
    kmeans.fit(all_features)

    vocab = kmeans.cluster_centers_
    return vocab

def get_sift_features(sift, vocab, vocab_size, img):
    img8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    _, descriptors = sift.detectAndCompute(img8bit, None)
    dists = pairwise_distances(vocab, descriptors)
    argmin = np.argmin(dists, 0)
    hist = np.bincount(argmin, minlength=vocab_size)
    features = hist / hist.sum()
    return features

def save_vocabulary(sift, images, vocab_size):
    vocab_filename = 'vocab.pkl'
    if not os.path.isfile(vocab_filename):
        # Construct the vocabulary
        vocab = build_vocabulary(sift, images, vocab_size)
        with open(vocab_filename, 'wb') as f:
            pickle.dump(vocab, f)
            print('{:s} saved'.format(vocab_filename))

def load_vocabulary(vocab_filename):
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab