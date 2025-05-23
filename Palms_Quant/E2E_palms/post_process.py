import scipy
import skimage.morphology
import numpy as np
np.bool = bool

#CS codes
CLASS_TO_SS = {"mauritia":-128, "euterpe":-96, "oenocarpus":-64} #Mask pixel value per class
CLASS_TO_CITYSCAPES = {"mauritia":15, "euterpe":25, "oenocarpus":35} #Output values

# Level 1: 0 to 2 pixels from boundary, for classes with many small obj. Level 2: 3 to 4 pixels from boundary. 
THRESHOLD = {"mauritia":3, "euterpe":1, "oenocarpus":2} 
#MIN_SIZE = {"mauritia":920, "euterpe":400, "oenocarpus":2320} #Minimum number of pixels per class (Based on the stats of size of crowns from field data, values for 5cm res)
MIN_SIZE = {"mauritia":500, "euterpe":400, "oenocarpus":200} #Minimum number of pixels per class (Based on the values after erosion) #MDD Mflexuosa:10 Eprecatoria:15 Obatahua:200 #Loreto mauritia":500, "euterpe":400, "oenocarpus":200 , ALP60mauritia":10, "euterpe":1, "oenocarpus":200  PRN PIU "mauritia":10, "euterpe":3, "oenocarpus":200, VAP01 "mauritia":500, "euterpe":400, "oenocarpus":1000 , ALP02 "mauritia":500, "euterpe":5, "oenocarpus":1100
'''
SELEM = {1: (np.ones((3,3))).astype(np.bool),
         2: (np.ones((5,5))).astype(np.bool),}
'''
SELEM = {3: (np.ones((3,3))).astype(np.bool), #MDD:15Brig,5Sando, LO:3, higher density, higher erosion
         1: (np.ones((1,1))).astype(np.bool), #15
		 2: (np.ones((1,1))).astype(np.bool),}

SELEN = {3: (np.ones((36,3))).astype(np.bool), #30Sandoval
         1: (np.ones((7,7))).astype(np.bool), 
		 2: (np.ones((3,3))).astype(np.bool),} 

		 
def watershed_cut(depthImage, ssMask):
    '''
    ssMask = ssMask.astype(np.int32)
    '''
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)

    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)

        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccLabels = skimage.morphology.label(ccImage) #labels instances
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000) #area_threshold indicates the max area to fill
        #ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])  #remove instances smaller than the average crown size
		
        ccIDs = np.unique(ccLabels)[1:]
        #print("ccIDs",ccIDs)
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            #ccIDMask = skimage.morphology.binary_dilation(ccIDMask, SELEM[THRESHOLD[semClass]]) #Dilation highlights bright parts and reduces darker ones
            ccIDMask = skimage.morphology.binary_erosion(ccIDMask, SELEM[THRESHOLD[semClass]]) #Erosion shrinks bright regions and enlarges dark regions
            ccIDMask = scipy.ndimage.binary_erosion(ccIDMask,SELEN[THRESHOLD[semClass]]) #structure erosion
            #ccIDMask = scipy.ndimage.binary_erosion(ccImage,(np.ones((10,10))).astype(np.bool),3)
            #resultImage[ccIDMask] = instanceID
            resultImage[ccIDMask] = csCode

    resultImage = resultImage.astype(np.uint8)
    return resultImage

def process_instances_raster(raster):
    
    resultImage = np.zeros(shape=raster.shape, dtype=np.float32)
    ninstances={"mauritia":0, "euterpe":0, "oenocarpus":0}
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ccImage = (raster == csCode)
       
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass]) #remove instances smaller than the average crown size
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000) #fill holes caused by the erosion
        ccLabels = skimage.morphology.label(ccImage)
            
        ccIDs = np.unique(ccLabels)[1:]
        #print("ccIDs",ccIDs)
        ninstances[semClass]=len(ccIDs)
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            #ccIDMask = skimage.morphology.binary_dilation(ccIDMask, SELEM[THRESHOLD[semClass]])
            #ccIDMask = skimage.morphology.binary_erosion(ccIDMask, SELEM[THRESHOLD[semClass]])
            instanceID = 1000 * csCode + ccID
            #resultImage[ccIDMask] = instanceID
            resultImage[ccIDMask] = csCode

    resultImage = resultImage.astype(np.uint16)
    return resultImage,ninstances





