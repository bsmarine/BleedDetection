import csv
import re
import numpy as np
import SimpleITK as sitk

def extract_coordinates(file):

    csvfile = open(file,'r')
    reader_iter = csv.reader(csvfile,delimiter=',')
    
    fiducials_coord = list()
    
    for row in reader_iter:
      #Find row whose first column is a Fiducial MRML node
      match = re.search('vtkMRMLMarkupsFiducialNode.*',row[0])
      if match == None:
        pass
      else:
        if match.end() == len(row[0]):
          #Extract x,y,z and package them in a list
          fiducials_coord.append([row[1],row[2],row[3]])
    
    #print ("Extracted Fiducial Coordinages "+str(fiducials_coord))

    return fiducials_coord

def extract_indices_annotation(coordinates,image):

    annotation_index = list()

    img = sitk.ReadImage(image)
    cor_arr = np.asarray(coordinates)
    cor_arr = cor_arr.astype(np.float)

    for point in cor_arr:
    #multiple by -1 x and y to convert RAS (Slicer) to LPS (nifti format)
      point[0] = point[0]*-1
      point[1] = point[1]*-1
      annotation_index.append(img.TransformPhysicalPointToIndex(point))
    
    return annotation_index