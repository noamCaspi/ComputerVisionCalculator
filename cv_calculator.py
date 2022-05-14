import cv2 as cv
import HandTrackingModule as htm
import numpy as np
import matplotlib.pyplot as plt

MULTIPLY = 10
DIVISION = 11
EQUALS = 12
PLUS = 13
MINUS = 14


BUTTONS_LOCATIONS = {0: [(250, 310), (305, 350)], 1: [(250, 310), (220, 270)],
                     2: [(350, 390), (220, 270)], 3: [(430, 470), (220, 270)],
                     4: [(250, 310), (140, 170)], 5: [(350, 390), (140, 170)],
                     6: [(430, 470), (140, 170)], 7: [(290, 310), (70, 100)],
                     8: [(365, 400), (70, 100)], 9: [(430, 470), (70, 100)],
                     MULTIPLY: [(340, 380), (305, 350)], DIVISION: [(430, 470), (305, 350)],
                     EQUALS: [(500, 540), (270, 350)], PLUS: [(500, 540), (60, 140)],
                     MINUS: [(500, 540), (170, 260)]}
THRESHOLD = 100


def compute(sol, operator, num):
    """
    Computes the solution of the calculator
    :param sol: the previous solution
    :param operator: the operator to apply
    :param num: the user's new input
    :return: the solution - <sol> <operator> <num>
    """
    if operator == MULTIPLY:
        return sol * num
    if operator == DIVISION:
        return 'math error' if num == 0 else sol / num
    if operator == EQUALS:
        return num
    if operator == PLUS:
        return sol + num
    if operator == MINUS:
        return sol - num


def alignImages(im1, im2):
    """
    Finds Homography between the images.
    :param im1: the source image
    :param im2: the destination image
    :return: the homography between im1 and im2
    """
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors (Oriented FAST and rotated BRIEF) - alternative for SIFT.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)
    return h


def initialize():
    """
    Starts capturing the video, and compute the homography between the calculator
    that is in the image to the reference.
    """
    # configure the camera:
    wCam, hCam = 640, 480
    cap = cv.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(detectionCon=0.75)
    # gives the camera time to focus
    for i in range(5):
        cap.read()
    success, src = cap.read()
    dst = cv.imread('reference.jpg', -1)
    H = alignImages(src, dst)
    return cap, detector, H


def find_pressed_button(new_point):
    """
    Given a point in the image, finds which of the button is the
    closest to that point (which is the button that was pressed).
    :param new_point: point in the image
    :return: a number, indicates the button that was pressed
    """
    min_key, min_dist = None, None
    for key, value in BUTTONS_LOCATIONS.items():
        dist = np.linalg.norm(new_point[:2] - (np.average(value[0]), np.average(value[1])))
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_key = key
    return min_key


def film_and_compute(cap, detector, H):
    """
    Tracks the user's presses on the calculator, and computes the result of the user's actions.
    :param cap: object of the camera's video
    :param detector: object that detects the user's hand
    :param H: homography matrix for the calculations
    :return: the result of the user's presses
    """
    down = True
    location = [0, 0]
    sol, num = 0, 0
    operator = EQUALS
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        cv.putText(img, f"num: {num}     sol: {sol}", (45, 375),
                   cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if len(lmList) != 0:
            # there is a hand in the image
            if down:
                if location[1] < lmList[8][2]:
                    # the finger is going down
                    location = lmList[8][1:]
                elif location[1] - THRESHOLD > lmList[8][2]:
                    # the finger started to go up -> there was a press
                    new_point = np.array([location[0], location[1], 1]) @ H.T
                    new_point /= float(new_point[2])
                    min_key = find_pressed_button(new_point)
                    if min_key < 10:
                        # this is a number
                        num = num * 10 + min_key
                    else:
                        sol = compute(sol, operator, num)
                        if sol == 'math error' or min_key == EQUALS:
                            break
                        num = 0
                        operator = min_key
                    down = False
            else:
                if location[1] > lmList[8][2]:
                    # the finger is higher than the saved location
                    location = lmList[8][1:]
                if location[1] + THRESHOLD < lmList[8][2]:
                    # the finger is significantly lower than 'location' so we go to 'down' mode
                    down = True
        cv.imshow("Image", img)
        cv.waitKey(1)
    return sol


if __name__ == "__main__":
    cap, detector, H = initialize()
    sol = film_and_compute(cap, detector, H)

    result_img = np.ones((500, 500))
    plt.figure()
    plt.imshow(result_img)
    plt.text(0, 250, f'sol: {sol}', size=30, color='white')
    plt.show()

