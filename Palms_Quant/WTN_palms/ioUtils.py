import numpy as np
import scipy.io as sio

np.random.seed(0)

#VGG_MEAN = [103.939, 116.779, 123.68]
VGG_MEAN = [107.687,132.034 ,77.097] #fron lista filtrada que tienen al menos un clase

CLASS_TO_SS = {"mauritia":1, "otra":2, "laotra":3}


def read_mat(path):
    return np.load(path)


def write_mat(path, m):
    np.save(path, m)


def read_ids(path):
    return [line.rstrip('\n') for line in open(path)]


class Batch_Feeder:
    def __init__(self, dataset, indices, train, batchSize, padWidth, padHeight, flip=False, keepEmpty=False):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._dataset = dataset
        self._indices = indices
        self._train = train
        self._batchSize = batchSize
        self._padWidth = padWidth
        self._padHeight = padHeight
        self._flip = flip
        self._keepEmpty = keepEmpty

    def set_paths(self, idList=None, gtDir=None, ssDir=None):
        self._paths = []

        if self._train:
            for id in idList:
                id =id.strip()
                if self._dataset == "kitti":
                    self._paths.append([id, gtDir+'/'+id+'.mat', ssDir+'/'+id+'.mat'])
                elif self._dataset == "cityscapes" or self._dataset == "pascal":
                    self._paths.append([id,
                                        ssDir + '/' + id + '_response.npz',
                                        gtDir + '/' + id + '_dirmap.npz',
                                        ssDir + '/' + id + '_deepmap.npz',
                                        ssDir + '/' + id + '_weightmap.npz'])
        else:
            for id in idList:
                if self._dataset == "kitti":
                    self._paths.append([id, ssDir+'/'+id+'.mat'])
                elif self._dataset == "cityscapes" or self._dataset == "pascal":
                    self._paths.append([id,
                                        ssDir + '/' + id + '_response.npz',
                                        gtDir + '/' + id + '_dirmap.npz'])

        self._numData = len(self._paths)

        assert self._batchSize < self._numData

    def shuffle(self):
        np.random.shuffle(self._paths)

    def next_batch(self):

        idBatch = []
        dirBatch = []
        gtBatch = []
        ssBatch = []
        ssMaskBatch = []
        weightBatch = []

        if self._train:
            while (len(idBatch) < self._batchSize):
                
                npz=np.load(self._paths[self._index_in_epoch][1])#segmetacion
                ss = (npz.f.arr_0).astype(float)
                ss[ss==-9999]=0
                ss, ssMask = self.ssProcess(ss[:,:,0])

                if ss.sum() > 0 or self._keepEmpty:
                    idBatch.append(self._paths[self._index_in_epoch][0])

                    npz=np.load(self._paths[self._index_in_epoch][2])#Mapa de direccion
                    dir = (npz.f.arr_0).astype(float)
                    npz=np.load(self._paths[self._index_in_epoch][3])#Mapa de profundidad
                    gt = (npz.f.arr_0).astype(float)
                    npz=np.load(self._paths[self._index_in_epoch][4])#pesos por tiled
                    weight =  (npz.f.arr_0[:,:,0]).astype(float)

                    dirBatch.append(self.pad(dir))
                    gtBatch.append(self.pad(gt))
                    weightBatch.append(self.pad(weight))
                    ssBatch.append(ss)

                self._index_in_epoch += 1

                if self._index_in_epoch == self._numData:
                    self._index_in_epoch = 0
                    self.shuffle()

            dirBatch = np.array(dirBatch)
            gtBatch = np.array(gtBatch)
            ssBatch = np.array(ssBatch)
            weightBatch = np.array(weightBatch)

            if self._flip and np.random.uniform() > 0.5:
                for i in range(len(dirBatch)):
                    for j in range(2):
                        dirBatch[i,:,:,j] = np.fliplr(dirBatch[i,:,:,j])
                    dirBatch[i, :, :, 0] = -1 * dirBatch[i, :, :, 0]
                    ssBatch[i] = np.fliplr(ssBatch[i])
                    gtBatch[i] = np.fliplr(gtBatch[i])
                    weightBatch[i] = np.fliplr(weightBatch[i])
            return dirBatch, gtBatch, weightBatch, ssBatch, idBatch
        else:
            for example in self._paths[self._index_in_epoch:min(self._index_in_epoch + self._batchSize, self._numData)]:
                npz=np.load(example[2])#Mapa de direccion
                dir = (npz.f.arr_0).astype(float)
                dirBatch.append(self.pad(dir))
                
                idBatch.append(example[0])

                npz=np.load(example[1])#segmetacion
                ss = (npz.f.arr_0).astype(float)
                ss[ss==-9999]=0
                ss, ssMask = self.ssProcess(ss[:,:,0])          
                ssBatch.append(self.pad(ss))
                ssMaskBatch.append(self.pad(ssMask))
            # imageBatch = np.array(imageBatch)
            dirBatch = np.array(dirBatch)
            ssBatch = np.array(ssBatch)
            ssMaskBatch = np.array(ssMaskBatch)
            # return imageBatch, dirBatch, ssBatch, idBatch
            self._index_in_epoch += self._batchSize
            return dirBatch, ssBatch, idBatch,ssMaskBatch

    def total_samples(self):
        return self._numData
 
    def image_scaling(self, rgb_scaled):
        # if self._dataset == "cityscapes":
        #     rgb_scaled = skimage.transform.pyramid_reduce(rgb_scaled, sigma=0.001)
            #rgb_scaled = skimage.transform.rescale(rgb_scaled, 0.5)

        rgb_scaled[:,:,0] = (rgb_scaled[:,:,0] - VGG_MEAN[0])
        rgb_scaled[:,:,1] = (rgb_scaled[:,:,1] - VGG_MEAN[1])
        rgb_scaled[:,:,2] = (rgb_scaled[:,:,2] - VGG_MEAN[2])

        return rgb_scaled
        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        #bgr = tf.concat(3, [
        #    blue - VGG_MEAN[0],
        #    green - VGG_MEAN[1],
        #    red - VGG_MEAN[2],
        #])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    def pad(self, data):
        if self._padHeight and self._padWidth:
            if data.ndim == 3:
                npad = ((0,self._padHeight-data.shape[0]),(0,self._padWidth-data.shape[1]),(0,0))
            elif data.ndim == 2:
                npad = ((0, self._padHeight - data.shape[0]), (0, self._padWidth - data.shape[1]))
            padData = np.pad(data, npad, mode='constant', constant_values=0)

        else:
            padData = data

        return padData

    def ssProcess(self,ssImage):
        ssMask = np.zeros(shape=ssImage.shape, dtype=np.float32)
        ssImageInt = ssImage
        #ssMask = ssImage
        if ssImageInt.dtype == np.float32:
            ssImageInt = (ssImageInt*255).astype(np.uint8)

        # order: Person, Rider, Motorcycle, Bicycle, Car, Truck, Bus, Train

        ssMask += (ssImageInt==CLASS_TO_SS['mauritia']).astype(np.float32)*1
        ssMask += (ssImageInt==CLASS_TO_SS['otra']).astype(np.float32)*2
        ssMask += (ssImageInt==CLASS_TO_SS['laotra']).astype(np.float32)*3


        ssBinary = (ssMask != 0).astype(np.float32)

        #ssMask[ssMask == 0] = 1 # temp fix

        ssMask = (ssMask - 5) * 32

        return ssBinary, ssMask
