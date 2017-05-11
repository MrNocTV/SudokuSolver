import cv2 
import numpy as np
import copy

def get_sudoku_board(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )

    # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('thresh', 600, 600)
    # cv2.imshow('thresh', thresh)

    biggest_contour, biggest_polygon = get_biggest_contour(thresh)
    if biggest_contour is not None:
        # cv2.drawContours(img, [biggest_contour], 0, (0, 255, 0), 2)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', 600, 600)
        # cv2.imshow('img', img)
        sudoku_board = perspective_transform(biggest_polygon, img)
        print(biggest_polygon)
        
        # extract_number(sudoku_board)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         cv2.destroyAllWindows()
        #         break
        return sudoku_board
    return None
    

def get_biggest_contour(thresh_img):
    img, contours, hier = cv2.findContours(
        thresh_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    biggest_contour = None 
    biggest_polygon = None 
    max_area = -1
    img_area = img.shape[0] * img.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= img_area/6:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_poly) == 4 and area > max_area:
                biggest_contour = contour 
                biggest_polygon = approx_poly
                max_area = area
    return biggest_contour, biggest_polygon

def extract_number(sudoku_board):
    cv2.imshow('board', sudoku_board)
    gray = cv2.cvtColor(sudoku_board, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.fastNlMeansDenoising(gray, 10, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )
    cv2.imshow('processed', thresh)
    thresh2 = copy.copy(thresh)
    image, contours, hier = cv2.findContours(
        thresh2,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    numbers = []

    for contour in contours:
        if 48 < cv2.contourArea(contour) < 400:
            x,y,w,h = cv2.boundingRect(contour)
            if h > 18:
                epsilon = 0.01*cv2.arcLength(contour, True)
                approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                cv2.rectangle(sudoku_board, (x-5,y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
                # get region of interest 
                roi = thresh[y-5:y+h+5, x-5:x+w+5]
                # convert to 10x10 pixel 
                roismall = cv2.resize(roi, (10, 10))
                numbers.append((roi, (x,y)))
    cv2.imshow('b', sudoku_board)
    return numbers


def perspective_transform(biggest_polygon, img):
    pst1 = np.float32(biggest_polygon)
    pst2 = np.float32([[0, 0], [0, 430], [430, 430], [430, 0]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    board = cv2.warpPerspective(img, M, (430, 430))
    return board

def predict(number, knn):
    number = cv2.resize(number, (10, 10))
    number = number.reshape((1, 100))
    number = np.float32(number)
    ret, results, neighbors, dist = knn.findNearest(number, 3)
    predicted_label = str(int(results[0][0]))
    return predicted_label
