import sys
from PyQt4.QtCore import QRect, Qt, QPoint
from PyQt4.QtGui import QApplication, QMainWindow, QLabel, qRgb, QImage, QPixmap, QFileDialog, QInputDialog
from ui_mainwindow import Ui_MainWindow
import numpy as np
import pyopencl as cl
from PIL import Image
from timeit import default_timer as timer

# ----------------------------------------------------------------------------------------------------------------------
# http://www.swharden.com/blog/2013-06-03-realtime-image-pixelmap-from-numpy-array-data-in-qt/

GREY_PALETTE = [qRgb(i, i, i) for i in range(256)]

# ----------------------------------------------------------------------------------------------------------------------

class NLMeans:
    def __init__(self, ctx):
        self.ctx = ctx
        self.queue = cl.CommandQueue(ctx)
        self.prg = None
        self.mask = None

        self.ax = 4
        self.sx = 2
        self.a = 1.0
        self.h = 1.0

        self._build_kernel()
        self._build_mask()

    def _build_kernel(self):
        with open("nlmeans.cl") as fp:
            KERNEL_CODE = fp.read()

        code = """#define AX {self.ax}
                  #define AY {self.ax}
                  #define SX {self.sx}
                  #define SY {self.sx}
                  """.format(self=self) + KERNEL_CODE
        self.prg = cl.Program(self.ctx, code).build()

    def _build_mask(self):
        n = 2*self.sx + 1
        c = n//2
        self.mask = np.empty((n,n)).astype(np.float32)

        for x in range(n):
            for y in range(n):
                self.mask[x,y] = np.exp(-((x-c)**2 + (y-c)**2) / self.a**2)

        self.mask /= self.mask.sum()

    def setAx(self, ax):
        self.ax = ax
        self._build_kernel()

    def setSx(self, sx):
        self.sx = sx
        self._build_kernel()
        self._build_mask()

    def setA(self, a):
        self.a = a
        self._build_mask()

    def setH(self, h):
        self.h = h

    def processImage(self, data):
        """array of float32 -> array of float32"""
        output = np.empty_like(data)
        rows, cols = data.shape

        mf = cl.mem_flags
        data_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=data)
        mask_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.mask)
        output_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, data.nbytes)

        args = [data_g, output_g, mask_g, np.int32(rows), np.int32(cols), np.float32(self.h)]
        #self.prg.NLMeans_kernel(self.queue, data.shape, None, *args)
        self.prg.NLMeans_kernel(self.queue, (256,), (256,), *args)
        cl.enqueue_copy(self.queue, output, output_g)

        return output

# ----------------------------------------------------------------------------------------------------------------------

def choose(parent, options, title, description):
    strings = ["%d. %r" % (i, item) for i, item in enumerate(options, 1)]
    selected, ok = QInputDialog.getItem(parent, title, description, strings, editable=False)
    if ok:
        return options[strings.index(selected)]
    else:
        raise RuntimeError("getItem failed")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.axSlider.valueChanged.connect(self.setAx)
        self.sxSlider.valueChanged.connect(self.setSx)
        self.aSlider.valueChanged.connect(lambda x: self.setA(x/1000))
        self.hSlider.valueChanged.connect(lambda x: self.setH(x/1000))

        self.showOriginal = False
        self.loadButton.clicked.connect(lambda: self.loadImage(QFileDialog.getOpenFileName(self, "Open Image")))
        self.saveButton.clicked.connect(lambda: self.saveImage(QFileDialog.getSaveFileName(self, "Save Image", filter="*.png")))
        self.toggleOriginalButton.clicked.connect(self.toggleImage)

        platform = choose(self, cl.get_platforms(), "OpenCL Platform", "Please choose OpenCL Platform")
        print(platform)
        device = choose(self, platform.get_devices(), "OpenCL Device", "Please choose OpenCL Device")
        print(device)
        ctx = cl.Context([device])
        self.nlmeans = NLMeans(ctx)

        self.imageLabel = QLabel()
        self.imageScrollarea.setWidget(self.imageLabel)
        self.maskScrollarea.setAlignment(Qt.AlignCenter)

        self.maskLabel = QLabel()
        self.maskScrollarea.setWidget(self.maskLabel)
        self.showMask()

        self.loadImage("lena.jpg")

        self.resetButton.clicked.connect(self.resetParameters)
        self.resetParameters()


    def setAx(self, ax, process=True):
        self.axSlider.setValue(ax)
        self.axSpinbox.setValue(ax)
        self.nlmeans.setAx(ax)
        if process: self.process()

    def setSx(self, sx, process=True):
        self.sxSlider.setValue(sx)
        self.sxSpinbox.setValue(sx)
        self.nlmeans.setSx(sx)
        self.showMask()
        if process: self.process()

    def setA(self, a, process=True):
        self.aSlider.setValue(a*1000)
        self.aSpinbox.setValue(a)
        self.nlmeans.setA(a)
        self.showMask()
        if process: self.process()

    def setH(self, h, process=True):
        self.hSlider.setValue(h*1000)
        self.hSpinbox.setValue(h)
        self.nlmeans.setH(h)
        if process: self.process()

    def resetParameters(self):
        self.setAx(4, False)
        self.setSx(2, False)
        self.setA(1.0, False)
        self.setH(20.0)

    def showMask(self):
        n = self.nlmeans.mask.shape[0]
        m = n*8

        mask = self.nlmeans.mask / self.nlmeans.mask[n//2,n//2] # rescale to make center extreme
        mask = ((1.0 - mask)*255.0).astype(np.uint8)

        maskImage = QImage(n, n, QImage.Format_Indexed8)
        maskImage.setColorTable(GREY_PALETTE)
        for i in range(n):
            for j in range(n):
                maskImage.setPixel(QPoint(i, j), mask[i,j])

        maskImage = maskImage.scaled(m, m, Qt.KeepAspectRatio)
        self.maskLabel.setPixmap(QPixmap.fromImage(maskImage))
        self.maskLabel.setGeometry(QRect(0, 0, m, m))


    def loadImage(self, path):
        if not path:
            return

        img = Image.open(path)
        self.originalData = np.asarray(img.convert("L")).astype(np.float32)
        self.data = self.originalData.copy()
        self.showImage()

    def showImage(self):
        if self.showOriginal:
            data = self.originalData.astype(np.uint8)
        else:
            data = self.data.astype(np.uint8)

        rows, cols = data.shape
        image = QImage(data, cols, rows, QImage.Format_Indexed8)
        image.setColorTable(GREY_PALETTE)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.imageLabel.setGeometry(QRect(0, 0, rows, cols))

    def saveImage(self, path):
        if not path:
            return

        im = Image.fromarray(self.data.astype(np.uint8))
        with open(path, mode="wb") as fp:
            im.save(fp, "png")

    def toggleImage(self):
        self.showOriginal ^= True
        self.showImage()

    def process(self):
        t0 = timer()
        self.data = self.nlmeans.processImage(self.originalData)
        mpx = self.data.shape[0] * self.data.shape[0] / 1e6
        t = timer() - t0
        self.showImage()
        self.timeSpinbox.setValue(t*1000)
        self.fpsSpinbox.setValue(1/t)
        self.mpixSpinbox.setValue(mpx/t)

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
