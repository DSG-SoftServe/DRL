# ---------------------------------------------------------------------#
#----------------------------------------------------------------------#


from numpy import *
import numpy as np
from numpy import dot, sqrt, diag
from PIL import Image, ImageOps, ImageFilter
import cPickle


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#


class SaveClass(object):
    def __init__(self, obj):
        self.obj = obj
        self.data = self.obj.getter()

    def picleSaver(self, folderName):
        f = file(folderName + str(self.data.shape).replace(" ", "") + ".txt", "wb")
        cPickle.dump(self.data, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        return self

    def pictureSaver(self, folderName):
        i = 0
        for pic in self.data:
            imsave = Image.fromarray(np.array(pic))
            imsave.save(folderName + str(i) + ".jpg", "JPEG", quality=100)
            i += 1
        return self


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#


class CutClass(object):
    REPORT = "OK"

    def __init__(self, img=0, array=np.array(0), xwindow=25, ywindow=25):
        self.imgArray = np.array(array)
        if img != 0:
            self.img = img
            self.imgArray = np.array(img)
        self.xwindow = xwindow
        self.ywindow = ywindow
        self.windowSet = []

    def cutter(self, conv=False, step=25):
        window = np.zeros(shape=(self.xwindow, self.ywindow))
        irange = range(self.imgArray.shape[0] / (self.xwindow * (not (conv)) + step * conv))
        jrange = range(self.imgArray.shape[1] / (self.ywindow * (not (conv)) + step * conv))
        for i in irange[:len(irange) - (self.xwindow / step - 1) * conv]:
            for j in jrange[:len(jrange) - (self.ywindow / step - 1) * conv]:
                window = self.imgArray[i * (self.xwindow * (not (conv)) + step * conv): i * (
                    self.xwindow * (not (conv)) + step * conv) + self.xwindow,
                         j * (self.ywindow * (not (conv)) + step * conv): j * (
                             self.ywindow * (not (conv)) + step * conv) + self.ywindow]
                self.windowSet.append(window)
        self.windowSet = np.array(self.windowSet)
        return self

    def getter(self):
        return self.windowSet

    def status(self):
        return str(self.windowSet.shape) + "\n" + self.REPORT


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#


class RandomCutClass(CutClass):
    def cutter(self, winNum):
        window = np.zeros(shape=(self.xwindow, self.ywindow))
        for i in range(winNum):
            xrand = np.random.randint(0, self.imgArray.shape[0] - self.xwindow)
            yrand = np.random.randint(0, self.imgArray.shape[1] - self.ywindow)
            window = self.imgArray[xrand: xrand + self.xwindow, yrand: yrand + self.ywindow]
            self.windowSet.append(window)
        self.windowSet = np.array(self.windowSet)
        return self


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#


class CutClassWindow(CutClass):
    def __init__(self, img=0, array=np.array(0), xy1=(25, 25), xy2=(29, 29)):
        self.imgArray = np.array(array)
        if img != 0:
            self.img = img
            self.imgArray = np.array(img)
        self.xy1 = xy1
        self.xy2 = xy2
        self.windowSet = []

    def cutter(self):
        window = np.zeros(shape=(self.xy2[0] - self.xy1[0], self.xy2[1] - self.xy1[1]))
        window = self.imgArray[self.xy1[0]: self.xy2[0], self.xy1[1]: self.xy2[1]]
        self.windowSet.append(window)
        self.windowSet = np.array(self.windowSet)
        return self


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#