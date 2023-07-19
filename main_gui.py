import threading
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageQt, ImageEnhance
from functools import partial

import trainer

class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, image):
        super(ImageViewer, self).__init__()

        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)

        self.image = QtGui.QPixmap.fromImage(image)
        self.item = QtWidgets.QGraphicsPixmapItem(self.image)
        self.scene.addItem(self.item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def change_image(self, image):
        self.scene.clear()
        self.image = QtGui.QPixmap.fromImage(image)
        self.item = QtWidgets.QGraphicsPixmapItem(self.image)
        self.scene.addItem(self.item)
        self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)
        self.scene.setSceneRect(self.item.boundingRect())
    
    def setImage(self, image):
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))

class LabeledImageViewer(QtWidgets.QWidget):
    def __init__(self, image, label, show_psnr=True):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(label)
        self.layout.addWidget(self.label)

        self.viewer = ImageViewer(image)
        self.layout.addWidget(self.viewer)

        if show_psnr:
            self.psnr_label = QtWidgets.QLabel("PSNR: N/A")
            self.layout.addWidget(self.psnr_label)
        else:
            self.psnr_label = None
    
    def setImage(self, image):
        self.viewer.change_image(image)

    def setPSNR(self, psnr):
        if self.psnr_label is not None:
            self.psnr_label.setText(f"PSNR: {psnr:.2f}")


class DemoWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg']

        self.initUI()

        # Fetch predicted image from the trainer every second
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000)

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout(self)

        self.image_path = 'image1.jpg'

        # Create horizontal layout for the image buttons
        self.button_layout = QtWidgets.QHBoxLayout()

        self.image_buttons = []
        for i in range(7):
            button = QtWidgets.QPushButton()
            # Create a QPixmap from the image file
            pixmap = QtGui.QPixmap(self.image_paths[i])
            # Scale the pixmap to the desired size while maintaining its aspect ratio
            pixmap = pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio)
            # Create an QIcon from the QPixmap
            button_icon = QtGui.QIcon(pixmap)
            button.setIcon(button_icon)
            button.setIconSize(pixmap.rect().size())
            button.setFixedSize(pixmap.rect().size())
            button.clicked.connect(partial(self.set_image_path, self.image_paths[i]))
            self.button_layout.addWidget(button)
            self.image_buttons.append(button)

        # Add the button layout to the main layout
        self.layout.addLayout(self.button_layout)

        self.hbox = QtWidgets.QHBoxLayout()

        self.image_size = (256, 256)

        # Load the original image
        self.pil_image = Image.open(self.image_path) # PIL Image
        self.original_image = ImageQt.ImageQt(self.pil_image) # QImage
        self.original_view = LabeledImageViewer(self.original_image, "Input Image", show_psnr=False)
        self.hbox.addWidget(self.original_view)

        # Load placeholder image
        self.placeholder_image = QtGui.QImage(self.image_path)

        self.mlp_trainer = trainer.Trainer(self.image_path, self.image_size)
        self.mlp_exp_trainer = trainer.Trainer(self.image_path, self.image_size, "mlp_exposure")

        # Display placeholder images for NeRF and RawNeRF
        self.nerf_view = LabeledImageViewer(self.placeholder_image, "Baseline MLP")
        self.exp_nerf_view = LabeledImageViewer(self.placeholder_image, "Correction MLP")

        self.train_button = QtWidgets.QPushButton("Train", self)
        self.train_button.clicked.connect(self.train)
        
        self.hbox.addWidget(self.nerf_view)
        self.hbox.addWidget(self.exp_nerf_view)

        # Add the hbox layout to the main layout
        self.layout.addLayout(self.hbox)

        # Create the exposure slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(-100, 100)  # Set the range to represent exposure levels
        self.slider.setValue(0)  # Set initial value to 0 (middle)
        self.slider.valueChanged[int].connect(self.change_exposure)

        # Add the slider to the main layout
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.train_button)

        self.setLayout(self.layout)
        self.setWindowTitle('Exposure correction')
        self.show()
    
    def train(self):
        print('Training the model...')

        # Signal the previous models to stop training
        if hasattr(self, 'mlp_trainer'):
            self.mlp_trainer.stop = True
        if hasattr(self, 'mlp_exp_trainer'):
            self.mlp_exp_trainer.stop = True
        
        # Normalize the exposure value in [1,2]
        exposure = self.slider.value()/100 * (2 - 1) + 1
        print(exposure)
        self.mlp_trainer = trainer.Trainer(self.image_path, self.image_size, exposure=exposure)
        self.mlp_exp_trainer = trainer.Trainer(self.image_path, self.image_size, "mlp_exposure", exposure=exposure)
        thread1 = threading.Thread(target=self.mlp_trainer.run, args=())
        thread2 = threading.Thread(target=self.mlp_exp_trainer.run, args=())
        thread1.start()
        thread2.start()

    def set_image_path(self, path):
        self.image_path = path

        # Recreate trainers with the new image path
        self.mlp_trainer = trainer.Trainer(self.image_path, self.image_size)
        self.mlp_exp_trainer = trainer.Trainer(self.image_path, self.image_size, "mlp_exposure")

        # Load the new original image
        self.pil_image = Image.open(self.image_path) # PIL Image
        self.original_image = ImageQt.ImageQt(self.pil_image) # QImage
        self.original_view.setImage(self.original_image)

        # Update placeholder images
        self.placeholder_image = QtGui.QImage(self.image_path)
        self.nerf_view.setImage(self.placeholder_image)
        self.exp_nerf_view.setImage(self.placeholder_image)

        # Reset the slider position to the middle
        self.slider.setValue(0)
    
    def update_image(self):
        if self.mlp_trainer.pred is not None:
            trainer_image = self.mlp_trainer.pred
            qimage = QtGui.QImage(trainer_image.data, trainer_image.shape[1], trainer_image.shape[0], QtGui.QImage.Format_RGB888)
            self.nerf_view.setImage(qimage)
            self.nerf_view.setPSNR(self.mlp_trainer.psnr)
        if self.mlp_exp_trainer.pred is not None:
            trainer_exp_image = self.mlp_exp_trainer.pred
            qimage_exp = QtGui.QImage(trainer_exp_image.data, trainer_exp_image.shape[1], trainer_exp_image.shape[0], QtGui.QImage.Format_RGB888)
            self.exp_nerf_view.setImage(qimage_exp)
            self.exp_nerf_view.setPSNR(self.mlp_exp_trainer.psnr)

    def change_exposure(self, value):
        # Adjust brightness of original image
        enhancer = ImageEnhance.Brightness(self.pil_image)
        new_image = enhancer.enhance(1.0 + value / 100.0)
        qimage = ImageQt.ImageQt(new_image)

        # Update the original image view
        self.original_view.viewer.change_image(qimage)


app = QtWidgets.QApplication([])
window = DemoWindow()
app.exec_()
