
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

cand_path = '../subset1/candidates.csv'
cands = readCSV(cand_path)


from generate import generate_view
# @profile
def slice(cands, voxelWidth=65):
    cubes = []
    labels = []
    
    files = set()
    curr = time.time()
    
    counter = 0
    num = 0

    for cand in cands:
        # load image
        img_path = '../subset1/' + cand[0] + '.mhd'

        try:
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
        except RuntimeError:
            continue

        if img_path not in files:
            files.add(img_path)
            print("adding new file", img_path)

        # new part: generate positives here
        if int(cand[4]) == 1:
            # print("found one positive")
            worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
            voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            
            pos = [voxelCoord[0]-voxelWidth/2, voxelCoord[0]+voxelWidth/2, voxelCoord[1]-voxelWidth/2, voxelCoord[1]+voxelWidth/2, voxelCoord[2]-voxelWidth/2, voxelCoord[2]+voxelWidth/2]
            pos = [int(i) for i in pos]
            patch = numpyImage[pos[0]:pos[1],pos[2]:pos[3],pos[4]:pos[5]]
            patch = normalizePlanes(patch)
            
            if patch.shape == (voxelWidth, voxelWidth, voxelWidth):               
                cubes.extend(generate_view(patch))
                if len(cubes) % 1000 == 0:
                    counter += 1
                    print("1000 positive generated, time", time.time() - curr)
                    curr = time.time()
                    np.save('positives-'+str(counter), cubes)


voxelWidth = 65
slice(cands, voxelWidth=65)

