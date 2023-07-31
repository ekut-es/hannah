# import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = "1"
import logging

import numpy as np
import random
from numba import cuda
import math
import numba
from scipy.special import erfinv
from numpy.random import default_rng



NF_SPLIT_FACTOR = 32


@cuda.jit(device=True)
def normalize(v):
    length = vector_length(v)
    return numba.float32(v[0]/length), numba.float32(v[1]/length), numba.float32(v[2]/length)


@cuda.jit(device=True)
def scalar(v, k):
    return numba.float32((v[0]*k[0])+(v[1]*k[1])+(v[2]*k[2]))


@cuda.jit(device=True)
def vector_length(v):
    return numba.float32(math.sqrt((v[0]*v[0])+(v[1]*v[1])+(v[2]*v[2])))


@cuda.jit(device=True)
def cross(v, k):
    z1 = (v[1]*k[2])-(v[2]*k[1])
    z2 = (v[2]*k[0])-(v[0]*k[2])
    z3 = (v[0]*k[1])-(v[1]*k[0])
    return numba.float32(z1), numba.float32(z2), numba.float32(z3)


@cuda.jit(device=True)
def add(v, k, l):
    return numba.float32(v[0]+k[0]+l[0]), numba.float32(v[1]+k[1]+l[1]), numba.float32(v[2]+k[2]+l[2])


@cuda.jit(device=True)
def mul(v, c):
    return numba.float32(v[0]*c), numba.float32(v[1]*c), numba.float32(v[2]*c)


@cuda.jit(device=True)
def rotate(v, k, angle):
    return add(mul(v, math.cos(angle)), mul(cross(v, k), math.sin(angle)), mul(k, scalar(k, v)*(1-math.cos(angle))))


@cuda.jit(device=True)
def trace(noisefilter, beam, splitIndex):
    index = int(((math.atan2(beam[1], beam[0]) * 180 / math.pi) + 360) * NF_SPLIT_FACTOR) % (360 * NF_SPLIT_FACTOR)
    for i in range(splitIndex[index], splitIndex[index + 1]):
        sphere = (noisefilter[i][0], noisefilter[i][1], noisefilter[i][2])
        beamDist = vector_length(beam)
        if beamDist < noisefilter[i][3]:
            return numba.float32(-1.0)
        length_beam_sphere = scalar(sphere, normalize(beam))
        if length_beam_sphere > 0.0:
            dist_beam_sphere = math.sqrt((noisefilter[i][3] * noisefilter[i][3]) - (length_beam_sphere * length_beam_sphere))
            if dist_beam_sphere < noisefilter[i][4]:
                return numba.float32(noisefilter[i][3])
    return numba.float32(-1.0)

@cuda.jit
def intersects(pointcloud, noisefilter, mostIntersectCount, mostIntersectDist, Intersections, Distances, DistanceCount, numPoints, splitIndex, intensity_factor):
    x, y = cuda.grid(2)
    numRays = 11
    # #------------ debug ------------------
    # k = cross((pointcloud[x][0], pointcloud[x][1], pointcloud[x][2]), (1.0, 2.0, 3.0))
    # x = cuda.threadIdx.x
    # bx = cuda.blockIdx.x
    # bdx = cuda.blockDim.x
    # if x == 1 and bx == 3:
    #     print(numba.typeof(vector_length((1.0, 2.0, 3.0))))
    #     print(numba.float32((pointcloud[100][0], pointcloud[100][1], pointcloud[100][2])))
    #     from pdb import set_trace;
    #     set_trace()
    #
    # #---------------------------------------

    if x < numPoints:
        original_point = (pointcloud[x][0], pointcloud[x][1], pointcloud[x][2])
        beam = (pointcloud[x][0], pointcloud[x][1], pointcloud[x][2])
        idx_count = 0
        # original point intersection
        intersection_dist = trace(noisefilter, beam, splitIndex)
        if intersection_dist > 0:
            #intersectCount[0] += 1
            Intersections[x][idx_count] = intersection_dist
            idx_count += 1

        # -------------------------------------------
        # ----------- rotated points ----------------
        divergenceAngle = 2e-4
        vectorRotationAngle = math.pi / 5
        numStreaks = 5
        numPointsPerStreak = 2
        z_axis = (numba.float32(0.0), numba.float32(0.0), numba.float32(1.0))
        rotVec = normalize(cross(beam, z_axis))
        for j in range(numStreaks):
            for k in range(1, numPointsPerStreak+1):
                if k <= numPointsPerStreak/2:
                    beam = rotate(original_point, rotVec, k*divergenceAngle)
                else:
                    beam = rotate(original_point, rotVec, (k-(numPointsPerStreak/2))*(-divergenceAngle))
                intersection_dist = trace(noisefilter, beam, splitIndex)
                if intersection_dist > 0:
                    Intersections[x][idx_count] = intersection_dist
                    idx_count += 1
                rotVec = rotate(rotVec, normalize(original_point), vectorRotationAngle)

        # --------------------------------------------
        # ------------ count intersections ------------
        N_intersects = 0

        for i, intersect in enumerate(Intersections[x]):
            if intersect != 0:
                N_intersects += 1
            for j in range(numRays):
                if intersect != 0:
                    if Distances[x][j] == 0:
                        DistanceCount[x][j] = 1
                        Distances[x][j] = intersect
                        break
                    elif intersect == Distances[x][j]:
                        DistanceCount[x][j] += 1
                        break

        R_all = N_intersects/numRays

        # --------------------------------------------
        # --------- find most intersected drop -------

        maxCount = 0
        maxIntersectionDist = numba.float32(0)
        for i, count in enumerate(DistanceCount[x]):
            if count > maxCount:
                maxCount = count
                maxIntersectionDist = Distances[x][i]
        mostIntersectCount[x] = maxCount
        mostIntersectDist[x] = maxIntersectionDist

        R_most = maxCount/N_intersects
        if R_all > 0.15:
            if R_most > 0.8:  # set point towards sensor
                dist = vector_length(pointcloud[x])
                pointcloud[x][0] *= maxIntersectionDist/dist
                pointcloud[x][1] *= maxIntersectionDist/dist
                pointcloud[x][2] *= maxIntersectionDist/dist
                pointcloud[x][3] *= 0.005
            else:  # delete point (filtered out later)
                pointcloud[x][0] = 0
                pointcloud[x][1] = 0
                pointcloud[x][2] = 0
                pointcloud[x][3] = 0
        else:  # modify intensity of unaltered point
            pointcloud[x][3] *= intensity_factor


# returns a point cloud as numpy array
def loadPointCloud(filepath, valuesPerPoint=4):
    frame = np.fromfile(filepath, dtype=np.float32)
    return np.reshape(frame, [int(np.size(frame)/valuesPerPoint), valuesPerPoint])


# returns a point cloud as cuda device array
def loadPointCloudToDevice(filepath, valuesPerPoint=4):
    frame = np.fromfile(filepath, dtype=np.float32)
    return cuda.to_device(np.reshape(frame, [int(np.size(frame)/valuesPerPoint), valuesPerPoint]))


def inverted_lognormal_cdf(D, R):
    return np.power(R, 0.23)*np.exp((np.sqrt(2)*np.log(1.43 - (0.0003*R))*erfinv((0.0116279*D)/(np.power(R, 0.22)))) - 0.328504)


def inverted_exponential_cdf(D, R):
    return -0.243902*np.power(R, 0.21)*np.log(0.0005124998718750320*D*np.power(R, -0.21))

def inverted_exponential_gm(D, R):
    return -0.436681*np.power(R, 0.48)*np.log(0.000916002564807181*D*np.power(R, 0.46)) - 5.9143581981431375


# use inverse transform sampling to sample particle sizes from arbitrary invertible distribution functions
def sample_particles(num_particles, precipitation, distribution='exponential'):
    if distribution == 'lognormal':
        sampling_func = inverted_lognormal_cdf
    if distribution == 'exponential':
        sampling_func = inverted_exponential_cdf
    if distribution == 'gm':
        sampling_func = inverted_exponential_gm

    return sampling_func(np.random.rand(num_particles), precipitation)*(1/2000)


def generateNoiseFilter(dim, dropsPerM3, precipitation=5.0, scale=1, distribution='exponential'):
    total_drops = int(abs(dim[0]-dim[1])*abs(dim[2]-dim[3])*abs(dim[4]-dim[5])*dropsPerM3)
    random.seed(42)
    x = np.random.uniform(dim[0], dim[1], total_drops)
    y = np.random.uniform(dim[2], dim[3], total_drops)
    z = np.random.uniform(dim[4], dim[5], total_drops)
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    size = sample_particles(total_drops, precipitation, distribution)*scale
    index = (((np.arctan2(y, x) * 180 / np.pi) + 360) * NF_SPLIT_FACTOR).astype(np.int32) % (360 * NF_SPLIT_FACTOR)
    nf = np.stack((x, y, z, dist, size, index), axis=-1)

    return sortNoiseFilter(nf)


def sortNoiseFilter(nf):
    split_index = np.zeros(360*NF_SPLIT_FACTOR + 1)
    nf = nf[nf[:, 3].argsort()]
    nf = nf[nf[:, 5].argsort(kind='mergesort')]
    for i in range(len(nf) - 1):
        if nf[i, 5] != nf[i+1, 5]:
            split_index[int(nf[i+1, 5])] = i + 1
    split_index[-1] = len(nf) - 1

    return nf, split_index


def cuda_trace(pc, nf, si, intensity_factor=0.9):
    num_points = np.shape(pc)[0]

    num_rays = 11
    nfd = cuda.to_device(nf)
    pcd = cuda.to_device(pc)
    splitIndex = cuda.to_device(si)
    threadsperblock = 32
    blockspergrid = (num_points + (threadsperblock - 1)) // threadsperblock
    Intersections = cuda.to_device(np.zeros([num_points, num_rays], dtype=float))
    Distances = cuda.to_device(np.zeros([num_points, num_rays], dtype=float))
    DistanceCount = cuda.to_device(np.zeros([num_points, num_rays], dtype=float))
    mostIntersectCount = cuda.to_device(np.zeros(num_points, dtype=float))
    mostIntersectDist = cuda.to_device(np.zeros(num_points, dtype=float))
    try:
        intersects[blockspergrid, threadsperblock](pcd, nfd, mostIntersectCount, mostIntersectDist, Intersections, Distances, DistanceCount, num_points, splitIndex, intensity_factor)
    except Exception as e:
        logging.info(str(e))
        logging.info('blockspergrid: '+str(blockspergrid))
        logging.info('threadsperblock: '+str(threadsperblock))
        logging.info('num_points: '+str(num_points))
        return pc

    result_pc = pcd.copy_to_host()
    result_pc = np.delete(result_pc, np.argwhere((result_pc[:, 0] == 0) & (result_pc[:, 1] == 0) & (result_pc[:, 2] == 0)), 0)
    return result_pc


if __name__ == '__main__':
    for i in range(100, 1300, 100):
        for j in range(1, 11):
            nf, si = generateNoiseFilter(np.array([0.0, 70.0, -40.0, 40.0, -2, 0.5], dtype=np.float32), i, precipitation=j, scale=1, distribution='gm')
            np.savez('/media/sven/7cbad348-a2fb-46e0-9461-651039557046/noisefilter_snow/nf_N=' + str(i) + '_R=' + str(j) + '.npz', nf=nf, si=si)
            print('saved nf_N=' + str(i) + '_R=' + str(j) + '.npz')
