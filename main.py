import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle
import os

import numpy as np
from scipy.signal import savgol_filter


def get_landmarks(path, type):
    """
    Analyse videos landmarks.

    :param path: path to video
    :return:
    """
    if os.path.exists(f'swings/{path}.pkl'):
        f = open(f'swings/{path}.pkl', 'rb')
        ld = pickle.load(f)
        f.close()
        return ld

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(f'swings/{path}.{type}')

    result = cv2.VideoWriter('swing.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (1080, 1920))

    landmarks = []

    while True:
        success, img = cap.read()

        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmark = []
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmark.append([lm.x, 1 - lm.y, lm.z, lm.visibility])
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            landmarks.append(landmark)

            img = cv2.resize(img, (1080, 1920), cv2.INTER_AREA)
            result.write(img)

            cv2.imshow("Image", img)

            cv2.waitKey(1)
        else:
            break

    f = open(f'swings/{path}.pkl', 'wb')
    pickle.dump(landmarks, f, -1)
    f.close()

    result.release()
    cap.release()
    cv2.destroyAllWindows()

    return landmarks


def get_weight_distribution(landmarks):
    landmark_names = mp.solutions.pose.PoseLandmark

    left = [landmark_names.LEFT_KNEE, landmark_names.LEFT_HIP]
    right = [landmark_names.RIGHT_KNEE, landmark_names.RIGHT_HIP]

    back_swing = savgol_filter(
        [landmark[landmark_names.LEFT_ELBOW][1] for landmark in landmarks[:int(len(landmarks) / 2)]], 100, 3)
    back_swing_top = np.argmax(back_swing)

    weight = []

    for landmark in landmarks:
        zero_heel = landmark[landmark_names.LEFT_HEEL][0]
        front_heel = landmark[landmark_names.RIGHT_HEEL][0]

        lefts = []
        rights = []

        for w in left:
            x = landmark[w][0]
            y = landmark[w][1]
            lefts.append((x - zero_heel) / (front_heel - zero_heel))

        for w in right:
            x = landmark[w][0]
            y = landmark[w][1]
            rights.append((x - zero_heel) / (front_heel - zero_heel))

        weight_perc = (sum(lefts) + sum(rights)) / (2 * len(lefts))

        weight.append(100 * (1 - min(1, weight_perc)))

    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)

    plt.plot(savgol_filter(weight, 100, 3), linewidth=3.0)
    plt.axvline(back_swing_top, color='k', linestyle='--', linewidth=0.7, label='Top of Swing')
    plt.xlabel('Frame')
    plt.ylabel('Weight Transfer Position')
    plt.title("Pro golf swing.")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    video = 'am_swing'
    video_landmarks = get_landmarks(video, 'mp4')
    get_weight_distribution(video_landmarks)
    #
    # video = 'w_swing'
    # video_landmarks = get_landmarks(video, 'mov')
    # get_weight_distribution(video_landmarks)
