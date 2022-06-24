import sys
import enum
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import resources
import res


class StyleHelper():
    @staticmethod
    def getWindowStyleSheet():
        return ("background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 rgba(44, 47, 51, 255), stop:1 rgba(35, 39, 42, 255));"
                "border-style: outset;"
                "border-width: 1px;"
                "border-color: qlineargradient( x1:0 y1:0, x2:1 y2:1, stop:0 rgba(16, 23, 38, 255), stop:1 rgba(24, 54, 83, 255));")
        # return ("background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 rgba(16, 23, 38, 255), stop:1 rgba(24, 54, 83, 255));"
        #         "border-style: outset;"
        #         "border-width: 1px;"
        #         "border-color: qlineargradient( x1:0 y1:0, x2:1 y2:1, stop:0 rgba(16, 23, 38, 255), stop:1 rgba(24, 54, 83, 255));")
                # "padding-left: 10px; padding-right: 10px; padding-top: 10px; padding-bottom: 10px;")

    @staticmethod
    def getTitleStyleSheet():
        return ("background-color: rgba(0,0,0,0);"
                "border: none;")
    @staticmethod
    def getLabelStyleSheet(fontSize):
        return ("color: white;"
                "background-color:  none;"
                "border: none;"
                # "padding-left: 10px; padding-right: 10px; padding-top: 1px; padding-bottom: 1px;"
                "font: " + str(fontSize) + "px 'Verdana';")

    @staticmethod
    def getPixmapStyleSheet():
        return ("background-color: none;"
                "border-style: solid;"
                "border-width: 4px;"
                "border-color: rgba(203, 192, 139, 100);")

    @staticmethod
    def getButtonStyleSheet(fontSize):
        return ("QPushButton {"
                "color: rgba(203,192,139,200);"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #40444B, stop:0.85 #40444B);"
                "border: none;"
                "padding-left: 10px; padding-right: 10px; padding-top: 1px; padding-bottom: 1px;"
                "font: " + str(fontSize) + "px'Verdana';"
                "font-weight: bold;"
                "text-align:left 10px;"
                "border-top-left-radius: 5px;"
                "border-top-right-radius: 5px;"
                "border-bottom-left-radius: 5px;"
                "border-bottom-right-radius: 5px;"
                "}"
                "QPushButton:hover {"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #40444B, stop:0.85 #858D9C);"
                "}"
                "QPushButton:disabled {"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 40444B, stop:0.85 rgba(35, 39, 45, 250));"
                "color: lightgray;"
                "}")

    @staticmethod
    def getTextButtonStyleSheet(fontSize):
        return ("QPushButton {"
                "color: rgb(168, 172, 155);"
                "background-color: transparent;"
                "border: none;"
                # "padding-left: 10px; padding-right: 10px; padding-top: 1px; padding-bottom: 1px;"
                "font: " + str(fontSize) + "px'Verdana';"
                "}"
                "QPushButton:hover {"
                "color: white;"
                "}")

    @staticmethod
    def getCloseStyleSheet():
        return  ("QToolButton { "
                 "image: url(:/xclose.png);"
                 "background-color: none; "
                 "border: none;"
                 "}"
                 "QToolButton:hover {"
                 "image: url(:/xclose_light.png); "
                 "}")

    @staticmethod
    def getMinimizeStyleSheet():
        return ("QToolButton { "
                "image: url(:/dash.png);"
                "background-color: none; "
                "border: none;"
                "}"
                "QToolButton:hover {"
                "image: url(:/dash_light.png); "
                "}")

    @staticmethod
    def getLineEditStyleSheet():
        return ("color: white;"
                "background-color: rgba(0, 0, 0, 0);"
                "border-style: solid;"
                "border-color: rgba(125, 133, 148, 140);"
                "border-width: 1px;"
                "font: 12px 'Verdana';")

    @staticmethod
    def getTabWidgetStyleSheet():
        return ("QTabWidget {border: none;}"
                "QTabWidget::pane {"
                "background-color: transparent;"
                "border-style: solid;"
                "border-width: 1px;"
                "border-color: rgba(69, 90, 128, 100);"
                "}"
                "QTabWidget::tab-bar {"
                "alignment: center;"
                "background-color: transparent;"
                "}"
                "QTabBar::tab {"
                "color: white;"
                "font: 12px 'Verdana';"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #687080, stop:0.85 #455a80);"
                "border-style: solid;"
                "border-color: rgba(125, 133, 148, 140);"
                "border-width: 1px;"
                "margin-left: 3px;"
                "margin-right: 3px;"
                "border-top-left-radius: 5px;"
                "border-top-right-radius: 5px;"
                "border-bottom-left-radius: 5px;"
                "border-bottom-right-radius: 5px;"
                "padding: 9px;"
                "min-width: 130px"
                "} "
                "QTabBar::tab:hover {"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #687080, stop:0.85 rgba(59,150,214,250));"
                "}"
                "QTabBar::tab:selected {"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #687080, stop:0.85 rgba(59,150,214,250));"#qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0 #687080, stop:0.85 rgba(114,150,214,250));"
                "padding: 10px;"
                "margin-bottom: -2px;"
                "}")

    @staticmethod
    def getTableStyleSheet():
        return ("QTableWidget {background-color: transparent; border: none; color: white; gridline-color: transparent;}"
                "QHeaderView::section {"
                "color: white;"
                "background-color: transparent;"
                "font: 12px 'Verdana';"
                "}"
                "QHeaderView {background-color: transparent;}"
                "QTableCornerButton::section {background-color: transparent;}"
                "QScrollBar:horizontal {"
                "    border: 1px solid rgb(32, 47, 130);"
                "    background-color: light grey;"
                "    height:12px;"
                "    margin: 0px 0px 0px 0px;"
                "}"
                "QScrollBar::handle:horizontal {"
                "    background: rgba(59,150,214, 160);"
                "    border-top-left-radius: 3px;"
                "    border-top-right-radius: 3px;"
                "    border-bottom-left-radius: 3px;"
                "    border-bottom-right-radius: 3px;"
                "    width: 100px;"
                "    min-height: 0px;"
                "}"
                "QScrollBar::add-line:horizontal {"
                "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "    stop: 0 rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130),  stop:1 rgb(32, 47, 130));"
                "    height: 0px;"
                "    subcontrol-position: bottom;"
                "    subcontrol-origin: margin;"
                "}"
                "QScrollBar::sub-line:horizontal {"
                "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "    stop: 0  rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130),  stop:1 rgb(32, 47, 130));"
                "    height: 0 px;"
                "    subcontrol-position: top;"
                "    subcontrol-origin: margin;"
                "}")

    @staticmethod
    def getSplitterStyleSheet():
        return ("QSplitter {background: transparent; border: none;}"
                "QSplitterHandle:hover {}"
                "QSplitter::handle:horizontal:hover {"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0.15 rgba(203,192,139,10), stop:0.5 rgba(203,192,139,200), stop:0.85 rgba(203,192,139,10));"
                "}"
               "QSplitter::handle:pressed{"
                "background-color: qlineargradient( spread:pad, x1:0 y1:0, x2:0 y2:1, stop:0.15 rgba(203,192,139,10), stop:0.5 rgba(203,192,139,200), stop:0.85 rgba(203,192,139,10));"
                "width: 10px;"
               "height: 500px;"
               "}")


class EventsButton(QToolButton):
    def __init__(self, mainSize, styleSheet, parent= None):
        super(EventsButton, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(25, 25)
        self.setMaximumSize(35, 35)
        self.setStyleSheet(styleSheet)


class TitleLabel(QLabel):
    def __init__(self, mainSize, parent= None):
        super(TitleLabel, self).__init__(parent)
        #self.setFixedSize(mainSize.width()*0.35, mainSize.height()*0.05)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.resize(mainSize.width()*0.35, mainSize.height()*0.05)
        self.image = QPixmap(":/logo.png")
        self.setPixmap(self.image.scaled(self.width(),self.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setStyleSheet(StyleHelper().getTitleStyleSheet())


class TextLabel(QLabel):
    def __init__(self, fontSize = 12, text = "", parent = None):
        super(TextLabel, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setText(text)
        scaleFactor = QDesktopWidget().availableGeometry().width() / 1920
        self.setStyleSheet(StyleHelper().getLabelStyleSheet(int(fontSize*scaleFactor)))


class CustomPixmap(QLabel):
    aspectRatio = 0
    def __init__(self, mainSize, image, parent= None):
        super(CustomPixmap, self).__init__(parent)
        # self.setFixedSize(mainSize.width()*0.95, mainSize.height()*0.47)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # policy.setHeightForWidth(True)
        self.setScaledContents(True)
        self.image = image
        self.aspectRatio = 3/4 #self.image.height() / self.image.width()
        self.setSizePolicy(policy)
        self.setMinimumSize(mainSize.width()*0.2, mainSize.width()*self.aspectRatio*0.2)
        self.setMaximumSize(mainSize.width(), mainSize.width()*self.aspectRatio)
        self.setStyleSheet(StyleHelper().getPixmapStyleSheet())
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setPixmap(self.image.scaled(self.width(),self.height(),
                      Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if(self.height() / self.width() > self.aspectRatio ):
            self.resize(self.width(), self.width()*self.aspectRatio)
        else:
            self.resize(self.height() / self.aspectRatio, self.height())


class PlainButton(QPushButton):
    def __init__(self,  mainSize, text = "", maxHeight = 0, parent = None):
        super(PlainButton, self).__init__(parent)
        if(maxHeight == 0):
            maxHeight = mainSize.height()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(mainSize.width() * 0.3, maxHeight* 0.05)
        self.setMaximumSize(mainSize.width() * 0.35, maxHeight * 0.05)
        self.setText(text)
        fontSize = 14
        scaleFactor = QDesktopWidget().availableGeometry().width() / 1920 - 0.05
        print(fontSize*scaleFactor)
        self.setStyleSheet(StyleHelper().getButtonStyleSheet(int(fontSize * scaleFactor)))


class MouseTypes(enum.Enum):
    Other = 0
    Top = 1
    Bottom = 2
    Left = 3
    Right = 4
    Move = 5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        QApplication.processEvents()
        self.mouseBtnPressed = MouseTypes.Other

    def init_ui(self):
        print("init ui")
        # flags & attributes
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        # geometry
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        desctopWidth = QDesktopWidget().availableGeometry().width()
        desctopHeight= QDesktopWidget().availableGeometry().height()
        self.setMinimumSize(desctopWidth / 2, desctopHeight / 2)
        self.setMaximumSize(desctopWidth , desctopHeight)
        self.resize(desctopWidth / 1.5, desctopHeight / 1.3)
        self.center()

        # general widgets
        self.MainWidget =  QWidget()
        self.interfaceWidget =  QWidget()

        # interface widgets
        self.title = TitleLabel(self.size()) # work with Qt-Resourse-File
        self.btnClose = EventsButton(self.size(), StyleHelper().getCloseStyleSheet())
        self.btnMinimize = EventsButton(self.size(), StyleHelper().getMinimizeStyleSheet(),)
        self.webcameraLabel = CustomPixmap(self.size(), QPixmap(":/stub1.png"),
            StyleHelper().getPixmapStyleSheet())
        self.visualizerLabel = CustomPixmap(self.size(), QPixmap(":/stub2.png"),
            StyleHelper().getPixmapStyleSheet())
        self.btnSelectImage = PlainButton(self.size(), "Select image", 700)
        self.btnWebCamReconstruct = PlainButton(self.size(), "Reconstruct from web-camera", 700)
        self.btnWebCamReconstruct.hide()
        #self.btnAddBook.resize(self.width()//3, self.height()//6)
        #style and effects
        self.shadowEffect = QGraphicsDropShadowEffect()
        self.shadowEffect.setBlurRadius(15)
        self.shadowEffect.setColor(QColor(0, 0, 0, 190))
        self.shadowEffect.setOffset(0)
        self.MainWidget.setStyleSheet(StyleHelper().getWindowStyleSheet())
        self.MainWidget.setGraphicsEffect(self.shadowEffect)

        # apply layouts to widgets
        self.set_layouts()
        self.setCentralWidget(self.interfaceWidget)
        self.centralWidget().setMouseTracking(True)

        self.btnClose.clicked.connect(self.close)
        self.btnMinimize.clicked.connect(self.showMinimized)
        self.mousePos = self.pos()
        print("end init ui")

    def set_layouts(self):
        # interface widget layout
        self.interfaceWidget.setLayout(QVBoxLayout())
        self.interfaceWidget.layout().addWidget(self.MainWidget)

        # layouts
        controlWH = QHBoxLayout()
        buttonsWH = QHBoxLayout()
        leftH = QHBoxLayout()
        leftV = QVBoxLayout()

        self.verticalWidgets = QVBoxLayout()

        # splitter
        bodyWH = QSplitter(Qt.Horizontal)
        bodyWH.setStyleSheet(StyleHelper.getSplitterStyleSheet())

        # splitter left and right widgets
        self.left = QWidget()
        self.left.setStyleSheet("background-color: transparent; border: none;")
        self.right = QWidget()
        self.right.setStyleSheet("background-color: transparent; border: none;")

        rightV = QVBoxLayout()
        rightV.addWidget(self.visualizerLabel)
        self.right.setLayout(rightV)

        # add widgets to layout
        controlWH.addWidget(self.title)
        controlWH.addStretch(1)
        controlWH.addWidget(self.btnMinimize)
        controlWH.addWidget(self.btnClose)

        leftV.addWidget(self.webcameraLabel)
        # leftV.addStretch(1)
        buttonsWH.addWidget(self.btnSelectImage)
        buttonsWH.addWidget(self.btnWebCamReconstruct)
        buttonsWH.addStretch(2)
        # leftH.addLayout(buttonsWV)
        # leftV.addLayout(leftH)
        self.left.setLayout(leftV)

        bodyWH.addWidget(self.left)
        bodyWH.addWidget(self.right)
        bodyWH.setCollapsible(0, False)
        bodyWH.setCollapsible(1, False)
        bodyWH.setStretchFactor(0, 1)
        bodyWH.setStretchFactor(1, 3)

        # buttonsWH.addWidget(self.btnGetRetBook)
        # buttonsWH.addWidget(self.btnAddBook)

        self.verticalWidgets.addLayout(controlWH)
        self.verticalWidgets.addWidget(bodyWH)
        self.verticalWidgets.addLayout(buttonsWH)
        #self.verticalWidgets.addLayout(buttonsWH)
        self.verticalWidgets.setContentsMargins(10, 10, 10, 20)
        self.MainWidget.setLayout(self.verticalWidgets)

    #move window to center
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def check_resizable_field(self, event):
        pos = event.globalPos()
        x = self.x()
        y = self.y()
        width = self.width()
        height = self.height()

        # define rectangles for resize
        rectTop = QRect(x+9, y, width - 18, 7)
        rectBottom = QRect(x+9, y+height-7, width -18, 7)
        rectLeft = QRect(x, y+9, 7 , height - 18)
        rectRight = QRect(x+width-7, y+ 9, 7, height -18)
        rectInterface = QRect(x+9, y+9, width - 18, height -18)
        if(rectTop.contains(pos)):
            self.setCursor(Qt.SizeVerCursor)
            return MouseTypes.Top
        elif (rectBottom.contains(pos)):
            self.setCursor(Qt.SizeVerCursor)
            return MouseTypes.Bottom
        elif (rectLeft.contains(pos)):
            self.setCursor(Qt.SizeHorCursor)
            return MouseTypes.Left
        elif (rectRight.contains(pos)):
            self.setCursor(Qt.SizeHorCursor)
            return MouseTypes.Right
        elif (rectInterface.contains(pos)):
            self.setCursor(QCursor())
            return MouseTypes.Move
        else:
            self.setCursor(QCursor())
            return MouseTypes.Other

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.mouseBtnPressed  =  MouseTypes.Other

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.mouseBtnPressed = self.check_resizable_field(event)
            self.mousePos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.mousePos)
        if (self.mouseBtnPressed == MouseTypes.Move):
            self.move(self.x() + delta.x(),self.y() + delta.y())
        elif (self.mouseBtnPressed == MouseTypes.Bottom):
            self.setGeometry(self.x(), self.y(),
                self.width(), self.height() + delta.y())
        elif (self.mouseBtnPressed == MouseTypes.Top
             and self.minimumHeight() <= self.height() - delta.y() <= self.maximumHeight()):
            self.setGeometry(self.x(), self.y() + delta.y(),
                self.width(), self.height() - delta.y())
        elif (self.mouseBtnPressed == MouseTypes.Left
              and self.minimumWidth() <= self.width() - delta.x() <= self.maximumWidth()):
            self.setGeometry(self.x() + delta.x(), self.y(),
                self.width() - delta.x(),self.height())
        elif (self.mouseBtnPressed == MouseTypes.Right):
            self.setGeometry(self.x(), self.y(),
                self.width() + delta.x(), self.height())
        else:
            self.check_resizable_field(event)
        self.mousePos = event.globalPos()