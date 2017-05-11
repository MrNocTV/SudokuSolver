import os 
import numpy as np
import cv2
import sys
import copy
from utils import get_sudoku_board, get_biggest_contour
from utils import perspective_transform

responses = []
keys = [number for number in range(48, 58)]
samples = np.empty([0, 100])
TRAIN_FOLDER = '../images/train-images/'
ESC = 27

def label_number_in_image(sudoku_board):
    global responses, keys, samples, ESC
    
    gray = cv2.cvtColor(sudoku_board, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.fastNlMeansDenoising(gray, 10, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )
    thresh2 = copy.copy(thresh)
    image, contours, hier = cv2.findContours(
        thresh2,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.imshow('thresh', thresh)
    for contour in contours:
        if 48 < cv2.contourArea(contour) < 400:
            x,y,w,h = cv2.boundingRect(contour)
            if h > 18:
                epsilon = 0.01*cv2.arcLength(contour, True)
                approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                cv2.rectangle(sudoku_board, (x-5,y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
                # get region of interest 
                roi = thresh[y-5:y+h+5, x-5:x+w+5]
                cv2.imshow('roi', roi)
                # convert to 10x10 pixel 
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow('board', sudoku_board)
                key = cv2.waitKey() & 0xFF
                print(key)
                if key == ESC:
                    sys.exit(0)
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape([1, 100])
                    samples = np.append(samples, sample, 0)
    return samples

def train_knn():
    samples = np.loadtxt('../data/samples.data', np.float32)
    responses = np.loadtxt('../data/responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model

def start_training():
    global samples, responses
    for filename in os.listdir(TRAIN_FOLDER):
        path = os.path.join(TRAIN_FOLDER, filename)
        sudoku_board = get_sudoku_board(path)
        if sudoku_board is not None:
            label_number_in_image(sudoku_board)
        else:
            print(filename)
    responses = np.array(responses, np.float32)
    responses = responses.reshape(responses.size, 1)
    print("Traning Complete")
    np.savetxt('../data/samples1.data', samples)
    np.savetxt('../data/responses1.data', responses)

if __name__ == '__main__':
    start_training()
    knn = train_knn()
