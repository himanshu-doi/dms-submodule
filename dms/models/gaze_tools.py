"""
Gaze estimation tools
"""

import math
import numpy as np
import cv2


def get_phi_theta_from_euler(euler_angles):
    return euler_angles[2], euler_angles[1]


def get_euler_from_phi_theta(phi, theta):
    return 0, -theta, -phi


def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def visualize_landmarks(image, landmarks):
    output_image = np.copy(image)
    for landmark in landmarks.reshape(-1, 2):
        cv2.circle(output_image, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
    return output_image


def limit_yaw(euler_angles_head):
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head

def visualize_headpose_result(face_image, nose_landmark, est_headpose):
    """Here, we take the original eye eye_image and overlay the estimated headpose."""
    output_image = np.copy(face_image)

    center_x = nose_landmark[0]
    center_y = nose_landmark[1]
    cv2.circle(output_image, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

    endpoint_x, endpoint_y = get_endpoint(est_headpose[1], est_headpose[0], center_x, center_y, 100)

    cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0), 3)
    return output_image


