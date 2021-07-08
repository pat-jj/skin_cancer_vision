import torch
import nltk
nltk.download('punkt')
import pyttsx3
import numpy as np
from PIL import Image
from torch.autograd import Variable
import os
import sys
from torchvision import transforms

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QMovie


class Ui_MainWindow(QWidget):
    UPLOADED = 0
    result = ''
    result_ResNext_101 = ''
    result_EfficientNet_B4 = ''
    result_ResNet_50 = ''

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Skin Cancer AI Doctor")
        MainWindow.resize(915, 691)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        # Uploaded Image
        self.labelimage = QtWidgets.QLabel(self.centralWidget)
        self.labelimage.setGeometry(QtCore.QRect(80, 480, 220, 220))
        self.labelimage.setText("")
        self.labelimage.setObjectName("labelimage")

        # Load Image button
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(690, 580, 90, 40))
        self.pushButton.setObjectName("pushButton")

        # Speech button
        self.pushButton_binary = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_binary.setGeometry(QtCore.QRect(95, 430, 90, 40))
        self.pushButton_binary.setObjectName("pushButton_binary")

        # Chat box (history)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setGeometry(QtCore.QRect(250, 70, 620, 360))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setFont(QFont('Arial', 14))

        # Input box
        self.typing = QtWidgets.QLabel(self.centralWidget)
        self.typing.setGeometry(QtCore.QRect(350, 460, 100, 20))
        self.typing.setFont(QFont('Arial', 15))
        self.typing.setText(u"Typing box")
        self.typing.setObjectName("typing")
        self.textEdit = QtWidgets.QTextEdit(self.centralWidget)
        self.textEdit.setGeometry(QtCore.QRect(350, 480, 520, 80))
        self.textEdit.setObjectName("textEdit")

        # Send button
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(800, 580, 70, 40))
        self.pushButton_2.setObjectName("pushButton_2")

        # Doctor GIF
        self.label_doctor = QtWidgets.QLabel(self.centralWidget)
        self.label_doctor.setGeometry(QtCore.QRect(40, 80, 200, 347))
        self.label_doctor.setObjectName("label_doctor")
        self.gif = QMovie('doctor-gif-1.gif')
        self.label_doctor.setMovie(self.gif)
        self.gif.start()

        # Title
        self.title_label = QtWidgets.QLabel(self.centralWidget)
        self.title_label.setGeometry(QtCore.QRect(250, 0, 400, 50))
        self.title_label.setFont(QFont('Arial', 30))
        self.title_label.setText(u"Skin Cancer AI Doctor")
        self.title_label.setObjectName("title_label")

        MainWindow.setCentralWidget(self.centralWidget)

        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 915, 17))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)

        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)

        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.retranslateUi(MainWindow)

        self.pushButton_2.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}
                                                QPushButton:hover{background:green;}''')
        self.pushButton_binary.setStyleSheet('''QPushButton{background:#F9F900;border-radius:5px;}
                                                QPushButton:hover{background:#C4C400;}''')
        self.pushButton.setStyleSheet('''QPushButton{background:#FFDAC8;border-radius:5px;}
                                                QPushButton:hover{background:#FFCBB3;}''')
        self.pushButton_binary.clicked.connect(self.speech_clicked)
        self.pushButton.clicked.connect(self.load_clicked)
        self.pushButton_2.clicked.connect(self.send_clicked)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def send_clicked(self):
        tmp = 'Patient (me): ' + self.textEdit.toPlainText()
        self.textEdit.clear()
        self.textEdit.setFocus()
        self.textBrowser.append(tmp)

        tokens = nltk.word_tokenize(tmp)
        if 'check' in tokens:
            ttmp = 'AI Doctor: ' + 'I am glad to help you!' + '\n' + \
                   'AI Doctor: ''Please upload your skin image.'
            self.textBrowser.append(ttmp)
            self.load_clicked()
        elif (('uploaded' in tokens) or ('result' in tokens)) and (self.UPLOADED == 1):
            self.speech_clicked()
            ttmp = 'AI Doctor: ' + self.result + '(Result by vote in speech)'+ '\n' + \
                   'Results by models: ' + '\n' + \
                   'ResNeXt-101:       ' + self.result_ResNext_101 + ' (Accuracy: 0.88219)' + '\n' + \
                   'EfficientNet-B4:     ' + self.result_EfficientNet_B4 + ' (Accuracy: 0.91208)' + '\n' + \
                   'ResNet-50:            ' + self.result_ResNet_50 + ' (Accuracy: 0.85314)'
            self.textBrowser.append(ttmp)
        elif (('uploaded' in tokens) or ('result' in tokens)) and (self.UPLOADED != 1):
            ttmp = 'AI Doctor: ' + 'Sure, but you have to upload your skin image first.' + '\n' + \
                   'AI Doctor: ' + 'You can do it by clicking "Load Image".'
            self.textBrowser.append(ttmp)
        elif ('hello' in tokens) or ('hi' in tokens) or ('Hello' in tokens) or ('Hi' in tokens):
            ttmp = 'AI Doctor: ' + 'Welcome! What can I do for you?'
            self.textBrowser.append(ttmp)
        elif ('thanks' in tokens) or ('thank' in tokens) or ('Thanks' in tokens) or ('Thank' in tokens):
            ttmp = 'AI Doctor: ' + 'You are welcome!'
            self.textBrowser.append(ttmp)
        elif ('another' in tokens) or ('other' in tokens) or ('again' in tokens) or ('and' in tokens):
            ttmp = 'AI Doctor: ' + 'No problem, please upload it.'
            self.textBrowser.append(ttmp)
        else:
            ttmp = 'AI Doctor: ' + 'Sorry, I don\'t understand your meaning'
            self.textBrowser.append(ttmp)

    def load_clicked(self):
        self.dir_choose = QFileDialog.getOpenFileName(self, "Choose a photo of your skin", os.getcwd())
        image = QImage(self.dir_choose[0])
        image = QPixmap(image)
        image = image.scaled(180, 180)
        self.labelimage.setPixmap(image)
        if (self.dir_choose[0]):
            self.UPLOADED = 1
            # print(type(self.dir_choose[0]))

    def speech_clicked(self):
        if(self.UPLOADED == 1):
            name = ['Actinic keratoses and intraepithelial carcinoma / Bowens disease',
                    'basal cell carcinoma', 'benign keratosis-like lesions',
                    'dermatofibroma', 'melanocytic nevi', 'vascular lesions', 'melanoma']

            model_ResNeXt_101 = torch.load("../skin_cancer/skin_cancer/model_ResNeXt_101.pkl", map_location=torch.device('cpu'))
            model_EfficientNet_B4 = torch.load("../skin_cancer/skin_cancer/model_EfficientNet_B4.pkl", map_location=torch.device('cpu'))
            model_ResNet_50 = torch.load("../skin_cancer/skin_cancer/model_ResNet_50.pkl", map_location=torch.device('cpu'))

            input_size = 224
            train_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.RandomRotation(20),
                                                  transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.7630374, 0.5456423, 0.5700383],
                                                                       [0.14092843, 0.15261276, 0.16997045])])

            img = Image.open(self.dir_choose[0])
            X = train_transform(img)
            X = X.unsqueeze(0)
            X = Variable(X)
            print("-" * 50)
            print(model_ResNeXt_101)
            print(model_EfficientNet_B4)
            print(model_ResNet_50)

            outputs = [100, 100, 100]
            predictions = [100, 100, 100]
            results = [100, 100, 100]
            for i in range(3):
                if i == 0:
                    outputs[i] = model_ResNeXt_101(X)
                elif i == 1:
                    outputs[i] = model_EfficientNet_B4(X)
                elif i == 2:
                    outputs[i] = model_ResNet_50(X)

                predictions[i] = outputs[i].max(1, keepdim=True)[1]
                results[i] = predictions[i].detach().numpy().squeeze()

            result = np.argmax(np.bincount(results))
            engine = pyttsx3.init()
            # engine.setProperty('gender', 'male')
            engine.say('It is' + name[result])
            self.result = name[result]
            self.result_ResNext_101 = name[results[0]]
            self.result_EfficientNet_B4 = name[results[1]]
            self.result_ResNet_50 = name[results[2]]
            engine.runAndWait()
        else:
            ttmp = 'AI Doctor: ' + 'Please upload your skin image first'
            self.textBrowser.append(ttmp)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Skin Cancer AI Doctor", "Skin Cancer AI Doctor"))
        self.pushButton.setText(_translate("Skin Cancer AI Doctor", "Load Image"))
        self.pushButton_binary.setText(_translate("Skin Cancer AI Doctor", "Speech"))
        self.pushButton_2.setText(_translate("Skin Cancer AI Doctor", "Send"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QMainWindow()
    w = Ui_MainWindow()
    w.setupUi(form)
    width = form.geometry().width()
    height = form.geometry().height()
    form.setFixedSize(width, height)
    form.setWindowIcon(QIcon(r'window_icon.png'))
    palette = QPalette()
    pix = QPixmap("background.png")
    palette.setBrush(QPalette.Background, QBrush(pix))
    form.setPalette(palette)
    form.show()
    sys.exit(app.exec_())

