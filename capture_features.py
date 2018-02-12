#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')
    test_num = 0
    models = []

    models.append([ \
        'beer',
        'bowl',
        'create',
        'disk_part',
        'hammer',
        'plastic_cup',
        'soda_can'])

    models.append([ \
        'biscuits',
        'soap',
        'soap2'])

    models.append([ \
        'biscuits',
        'soap',
        'book',
        'soap2',
        'glue'])

    models.append([ \
        'sticky_notes',
        'book',
        'snacks',
        'biscuits',
        'eraser',
        'soap2',
        'soap',
        'glue'])
    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models[test_num]:
        spawn_model(model_name)

        for i in range(200):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 200:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    print(model_name)
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()

    if test_num == 0:
        pickle.dump(labeled_features, open('training_set0.sav', 'wb'))
    elif test_num == 1:
        pickle.dump(labeled_features, open('training_set1.sav', 'wb'))
    elif test_num == 2:
        pickle.dump(labeled_features, open('training_set2.sav', 'wb'))
    elif test_num == 3:
        pickle.dump(labeled_features, open('training_set3.sav', 'wb'))

