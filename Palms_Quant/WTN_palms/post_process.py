import scipy.ndimage.interpolation
import scipy.misc
import skimage.morphology
import numpy as np

#CS codes: 24: person, 25: rider, 32: motorcycle, 33: bicycle, 26: car, 27: truck, 28: bus, 31: train
#PSP SS codes:
# CLASS_TO_SS = {"person":12, "rider":13, "motorcycle":18,
#                "bicycle":19, "car":13, "truck":15, "bus":16, "train":17}
CLASS_TO_SS = {"mauritia":-128, "otra":-96, "laotra":-64}
CLASS_TO_CITYSCAPES = {"mauritia":24, "otra":25, "laotra":32}
THRESHOLD = {"mauritia":2, "otra":1, "laotra":1}
MIN_SIZE = {"mauritia":100, "otra":20, "laotra":20}
SELEM = {1: (np.ones((3,3))).astype(np.bool),
         2: (np.ones((5,5))).astype(np.bool),
         3: (np.ones((7,7))).astype(np.bool),
        4: (np.ones((9,9))).astype(np.bool)}

def watershed_cut(depthImage, ssMask):
    ssMask = ssMask.astype(np.int32)
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)

    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)

        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
        ccImage = skimage.morphology.remove_small_holes(ccImage)
        ccLabels = skimage.morphology.label(ccImage)

        ccIDs = np.unique(ccLabels)[1:]
        for ccID in ccIDs:
            ccIDMask = (ccLabels == ccID)
            ccIDMask = skimage.morphology.binary_dilation(ccIDMask, SELEM[THRESHOLD[semClass]])
            instanceID = 1000 * csCode + ccID
            resultImage[ccIDMask] = instanceID

    resultImage = resultImage.astype(np.uint16)
    return resultImage









