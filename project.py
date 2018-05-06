
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
import time

from PIL import Image

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

cand_path = '../subset0/candidates.csv'
cands = readCSV(cand_path)

@profile
def slice(cands, voxelWidth=65):
    cubes = []
    labels = []
    
    files = set()
    curr = time.time()
    
    counter = 0
    num = 0

    for cand in cands[9000:]:
        # load image
        img_path = '../subset0/' + cand[0] + '.mhd'

        try:
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
        except RuntimeError:
            continue

        if img_path not in files:
            files.add(img_path)
            print("adding new file", img_path)

        worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
        
        pos = [voxelCoord[0]-voxelWidth/2, voxelCoord[0]+voxelWidth/2, voxelCoord[1]-voxelWidth/2, voxelCoord[1]+voxelWidth/2, voxelCoord[2]-voxelWidth/2, voxelCoord[2]+voxelWidth/2]
        pos = [int(i) for i in pos]
        patch = numpyImage[pos[0]:pos[1],pos[2]:pos[3],pos[4]:pos[5]]
        patch = normalizePlanes(patch)
        
        if patch.shape == (voxelWidth, voxelWidth, voxelWidth):
            counter += 1
            if counter % 1000 == 0:
                assert(len(cubes) == len(labels))
                # np.savez_compressed('cubes-'+str(num), cubes=cubes)
                # np.savez_compressed('labels-'+str(num), labels=labels)
                
                print(time.time()-curr, "seconds elapsed...")

                num += 1
                cubes = []
                labels = []
                curr = time.time()
                
            cubes.append(patch)
            labels.append(int(cand[4]))
    
    return cubes, labels


voxelWidth = 65
cubes, labels = slice(cands, voxelWidth=65)

