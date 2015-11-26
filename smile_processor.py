import argparse
import math

import numpy as np
import scipy
from scipy.io import loadmat
"""
Sample Usage:

    Run with default configurations:
    python simple_processor.py --keypoint_filepath  <path to .mat file>

"""
def parse_input():
    parser = argparse.ArgumentParser(description='Simple script to compute average lip curvature per time period')
    parser.add_argument('--keypoint_filepath', help = 'Path to Intraface keypoint .mat file', required=True)
    parser.add_argument('--min_year', help = 'Year to start buckets', default=1900)
    parser.add_argument('--max_year', help = 'Year to end buckets', default=2015)
    parser.add_argument('--bucket_size', help = 'Number of years in each time bucket', default=5)

    args = parser.parse_args()
    return args

class YearBucket(object):
    """
    This class handles bucketing each data point into a group of years.

    Years rounded down to the closest bucket. For example, if we had a
    1910 bucket and 1920 bucket, years 1910-1919 would map to the 1910 bucket,
    while years 1920-1929 years would map to the 1920 bucket.

    Sample Usage:
        YearBucket(1900, 2015, 5):
            bucket 0: 1900
            bucket 1: 1905
            bucket 2: 1910
            ...
    """

    def __init__(self, min, max, bucket_size=10):
        self.min = min
        self.max = max
        self.bucket_size = bucket_size
        self.buckets = {}

        year = min
        while year < max:
            bucket_key = self._get_bucket_key_for_year(year)
            self.buckets[bucket_key] = []
            year += bucket_size

    def _get_bucket_key_for_year(self, year):
        return int(year/self.bucket_size * self.bucket_size)

    def insert(self, year, lip_curvature):
        bucket_key = self._get_bucket_key_for_year(year)
        self.buckets.setdefault(bucket_key, []).append(lip_curvature)

    def get_buckets(self):
        return self.buckets

    def compute_average_per_bucket(self):
        result = {}
        for bucket_key, data_list in self.buckets.iteritems():
            result[bucket_key] = np.average(data_list)
        return result

def rad_to_angle(radians):
    return radians * 180/math.pi

def compute_lip_curvature(keypoints):
    """
    Compute average lip curvature in degrees.

    Lip curvature is calculated by taking the angles (a1,a2) between the
    ends of the lips (31, 37) with the top of the bottom lip (47)
    awesome ascii diagram where x and y are increasing in the
    direction of the arrows.

    ------------> x

    |    31 ------------ 37
    |      \ a1   |   a2/
    |       \     |    /
    | y      \    |   /
    |         \   |  /
    v          \  | /
                 47
    @param keypoints: np.array, 49x2 array of coordinates
    @return: float, lip curvature in degrees
    """
    left_lip_coord = keypoints[31]
    right_lip_coord = keypoints[37]
    top_of_bottom_lip = keypoints[47]

    # Compute left side
    x_dist = top_of_bottom_lip[0] - left_lip_coord[0]
    y_dist = top_of_bottom_lip[1] - left_lip_coord[1]
    left_angle = math.atan(float(y_dist)/x_dist)

    # Compute right side
    x_dist = right_lip_coord[0] - top_of_bottom_lip[0]
    y_dist = top_of_bottom_lip[1] - right_lip_coord[1]
    right_angle = math.atan(float(y_dist)/x_dist)

    avg_angle = (left_angle + right_angle)/2.0

    return rad_to_angle(avg_angle)


def main():
    args = parse_input()

    filepath = args.keypoint_filepath
    min_year = int(args.min_year)
    max_year = int(args.max_year)
    bucket_size = int(args.bucket_size)

    keypoint_data = scipy.io.loadmat(filepath)
    keypoint_mats = keypoint_data['scaled_rotated_faces'][0]

    male_buckets = YearBucket(min_year, max_year, bucket_size=bucket_size)
    female_buckets = YearBucket(min_year, max_year, bucket_size=bucket_size)

    for i in xrange(keypoint_mats.shape[0]):
        if i % 5000 == 0:
            print("Finished processing {0} entries".format(i))

        sample = keypoint_mats[i]

        keypoints = sample['keypoints']
        gender = str(sample['gender'][0])
        year = int(sample['year'])

        lip_curvature = compute_lip_curvature(keypoints)
        if gender.upper() == 'M':
            male_buckets.insert(year, lip_curvature)
        elif gender.upper() == 'F':
            female_buckets.insert(year, lip_curvature)
        else:
            print("Unexpected gender {0} for sample {1}".format(gender, i))

    print("Finished bucketing all samples by gender and years. Now computing averages...\n")

    male_bucket_averages = male_buckets.compute_average_per_bucket()
    female_bucket_averages = female_buckets.compute_average_per_bucket()

    print("Final averaged lip curvature (degrees) results:")
    print("male: {0}\n".format(male_bucket_averages))
    print("female: {0}\n".format(female_bucket_averages))

if __name__ == "__main__":
    main()
