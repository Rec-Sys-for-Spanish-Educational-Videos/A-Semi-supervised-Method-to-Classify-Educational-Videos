# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Arkin\Desktop\Valencia.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import numpy as np
import re
from extractArticles import WikipediaExtractor
import FinalVersion

class ExtractThread(QtCore.QThread):

 changeProgressBarMax =   QtCore.pyqtSignal(int)
 changeProgressBarValue = QtCore.pyqtSignal(int)
 changeExtractTextValue = QtCore.pyqtSignal(str)
 saveWikipedia = QtCore.pyqtSignal(list)
 
 
 def setInfo(self, maxDepth, lang, categories, classes):
     self.maxDepth = maxDepth
     self.lang = lang
     self.categories = categories
     self.classes = classes
     self.count = 0
 
 def callback(self, State, Title):
     self.count+=1
     self.changeProgressBarValue.emit(self.count)
     
     if(State == True):
         self.changeExtractTextValue.emit(Title + " has been extracted!\n")
     else:
         self.changeExtractTextValue.emit(Title + " could not be extracted!\n")
     
    
 def run(self):
     extractor =  WikipediaExtractor(self.maxDepth, self.lang)
     extractor.setCallbackFunction(self.callback)
   
     nr = 0
     self.changeExtractTextValue.emit("Calculating number of pages...\n")
     
     for i in range(0, len(self.categories)):
         nr+= extractor.findNrPagesinSubcategories(self.categories[i], self.classes[i])
         
     self.changeProgressBarMax.emit(nr)
     
     self.changeExtractTextValue.emit("Extracting the pages...\n")
     
     for i in range(0, len(self.categories)):
         extractor.findPagesinSubcategories(self.categories[i], self.classes[i])
     
    
     
     self.saveWikipedia.emit(extractor.labelledData)
    
     
 
    
        
class PredictThread(QtCore.QThread):


 changePredictTextValue = QtCore.pyqtSignal(str)
 
     
 def setInfo(self, wikipediaFile, threshold):
     self.wikipediaFile = wikipediaFile
     self.threshold = threshold
    

    
 def run(self):
     FinalVersion.setEmitFunc(self.changePredictTextValue)
     
     FinalVersion.predict( self.wikipediaFile, self.threshold)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 801, 571))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.languageLabel = QtWidgets.QLabel(self.tab)
        self.languageLabel.setGeometry(QtCore.QRect(60, 20, 51, 20))
        self.languageLabel.setObjectName("languageLabel")
        self.languageComboBox = QtWidgets.QComboBox(self.tab)
        self.languageComboBox.setGeometry(QtCore.QRect(160, 20, 69, 22))
        self.languageComboBox.setObjectName("languageComboBox")
        self.languageComboBox.addItem("")
        self.languageComboBox.addItem("")
        self.languageComboBox.addItem("")
        self.languageComboBox.addItem("")
        self.saveExcelButton = QtWidgets.QPushButton(self.tab)
        self.saveExcelButton.setEnabled(False)
        self.saveExcelButton.setGeometry(QtCore.QRect(60, 410, 75, 23))
        self.saveExcelButton.setObjectName("saveExcelButton")
        self.depthLabel = QtWidgets.QLabel(self.tab)
        self.depthLabel.setGeometry(QtCore.QRect(60, 60, 47, 13))
        self.depthLabel.setObjectName("depthLabel")
        self.extractProgressBar = QtWidgets.QProgressBar(self.tab)
        self.extractProgressBar.setGeometry(QtCore.QRect(60, 350, 261, 23))
        self.extractProgressBar.setProperty("value", 0)
        self.extractProgressBar.setObjectName("extractProgressBar")
        self.depthSpinBox = QtWidgets.QSpinBox(self.tab)
        self.depthSpinBox.setGeometry(QtCore.QRect(160, 50, 42, 22))
        self.depthSpinBox.setObjectName("depthSpinBox")
        self.categoriesTableWidget = QtWidgets.QTableWidget(self.tab)
        self.categoriesTableWidget.setGeometry(QtCore.QRect(60, 130, 241, 151))
        self.categoriesTableWidget.setObjectName("categoriesTableWidget")
        self.categoriesTableWidget.setColumnCount(2)
        
        
        self.depthSpinBox.setValue(1)
        
        try:
            f = open('categories.txt','r',encoding='utf-8')
            
            categories = []
            
            classes = []
            
            for line in f.readlines():
               
                categories.append(re.findall("\'(.*?)\'", line))
                classes.append(re.findall('[0-9]+', re.sub("\'(.*?)\' ","",line)))
                
            f.close()
            
            self.categoriesTableWidget.setRowCount(len(categories))
            
            
            for i in range(0, len(categories)):
                self.categoriesTableWidget.setItem(i,0,QtWidgets.QTableWidgetItem(categories[i][0]))
                self.categoriesTableWidget.setItem(i,1,QtWidgets.QTableWidgetItem(classes[i][0]))
        except IOError:
            self.categoriesTableWidget.setRowCount(10)
        
        
 
  
        
        item = QtWidgets.QTableWidgetItem()
        self.categoriesTableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.categoriesTableWidget.setHorizontalHeaderItem(1, item)
        self.categoriesLabel = QtWidgets.QLabel(self.tab)
        self.categoriesLabel.setGeometry(QtCore.QRect(60, 100, 61, 20))
        self.categoriesLabel.setObjectName("categoriesLabel")
        self.extractButton = QtWidgets.QPushButton(self.tab)
        self.extractButton.setGeometry(QtCore.QRect(60, 300, 75, 23))
        self.extractButton.setObjectName("extractButton")
        self.logLabel = QtWidgets.QLabel(self.tab)
        self.logLabel.setGeometry(QtCore.QRect(400, 10, 51, 20))
        self.logLabel.setObjectName("logLabel")
        self.extractTextBrowser = QtWidgets.QTextBrowser(self.tab)
        self.extractTextBrowser.setGeometry(QtCore.QRect(400, 30, 351, 381))
        self.extractTextBrowser.setObjectName("extractTextBrowser")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.loadWikipediaBtn = QtWidgets.QPushButton(self.tab_2)
        self.loadWikipediaBtn.setGeometry(QtCore.QRect(30, 20, 131, 23))
        self.loadWikipediaBtn.setObjectName("loadWikipediaBtn")
        self.trainButton = QtWidgets.QPushButton(self.tab_2)
        self.trainButton.setEnabled(False)
        self.trainButton.setGeometry(QtCore.QRect(20, 380, 75, 23))
        self.trainButton.setObjectName("trainButton")
        self.thresholdSpinBox = QtWidgets.QSpinBox(self.tab_2)
        self.thresholdSpinBox.setGeometry(QtCore.QRect(100, 60, 42, 22))
        self.thresholdSpinBox.setObjectName("thresholdSpinBox")
        self.labelThreshold = QtWidgets.QLabel(self.tab_2)
        self.labelThreshold.setGeometry(QtCore.QRect(30, 70, 47, 13))
        self.labelThreshold.setObjectName("labelThreshold")
        self.savePredictionButton = QtWidgets.QPushButton(self.tab_2)
        self.savePredictionButton.setEnabled(False)
        self.savePredictionButton.setGeometry(QtCore.QRect(20, 440, 101, 23))
        self.savePredictionButton.setObjectName("savePredictionButton")
        self.keywordsRadioButton = QtWidgets.QRadioButton(self.tab_2)
        self.keywordsRadioButton.setGeometry(QtCore.QRect(30, 110, 82, 17))
        self.keywordsRadioButton.setObjectName("keywordsRadioButton")
        self.titleRadioButton = QtWidgets.QRadioButton(self.tab_2)
        self.titleRadioButton.setGeometry(QtCore.QRect(120, 110, 82, 17))
        self.titleRadioButton.setObjectName("titleRadioButton")
        self.trainTextBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.trainTextBrowser.setGeometry(QtCore.QRect(20, 150, 256, 192))
        self.trainTextBrowser.setObjectName("trainTextBrowser")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Valencia Educational Video Prediciton"))
        self.languageLabel.setText(_translate("MainWindow", "Language:"))
        self.languageComboBox.setItemText(0, _translate("MainWindow", "es"))
        self.languageComboBox.setItemText(1, _translate("MainWindow", "en"))
        self.languageComboBox.setItemText(2, _translate("MainWindow", "de"))
        self.languageComboBox.setItemText(3, _translate("MainWindow", "fr"))
        self.saveExcelButton.setText(_translate("MainWindow", "Save to excel"))
        self.depthLabel.setText(_translate("MainWindow", "Depth:"))

        item = self.categoriesTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Category"))
        item = self.categoriesTableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Label"))
        self.categoriesLabel.setText(_translate("MainWindow", "Categories:"))
        self.extractButton.setText(_translate("MainWindow", "Extract"))
        self.extractButton.clicked.connect(self.extractOnPush)
        self.logLabel.setText(_translate("MainWindow", "Log:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Wikipedia Extractor"))
        self.loadWikipediaBtn.setText(_translate("MainWindow", "Load wikipedia articles"))
        self.loadWikipediaBtn.clicked.connect(self.loadOnPush)
        self.trainButton.setText(_translate("MainWindow", "Train"))
        self.trainButton.clicked.connect(self.predictOnPush)
        self.labelThreshold.setText(_translate("MainWindow", "Threshold"))
        self.savePredictionButton.setText(_translate("MainWindow", "Save predictions"))
        self.keywordsRadioButton.setText(_translate("MainWindow", "Keywords"))
        self.titleRadioButton.setText(_translate("MainWindow", "Title"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Video Prediction"))
        

        
        
    def extractOnPush(self):
        self.extractButton.setDisabled(True)
        
        self.t = ExtractThread()
        
        categories = []
        classes = []
        
        for i in range(0,self.categoriesTableWidget.rowCount()):
            categories.append(self.categoriesTableWidget.item(i,0).text())
            classes.append(self.categoriesTableWidget.item(i,1).text())
            
       
        
        self.t.setInfo(int(self.depthSpinBox.value()) ,self.languageComboBox.currentText(), categories, classes )
        
        self.t.changeExtractTextValue.connect(self.setExtractText)
        self.t.changeProgressBarMax.connect(self.setProgressMax)
        self.t.changeProgressBarValue.connect(self.setProgressVal)
        self.t.saveWikipedia.connect(self.saveWikipedia)
        self.t.start()
        
    def predictOnPush(self):
        self.pred = PredictThread()
        
        self.pred.setInfo(self.fileNamePred, self.thresholdSpinBox.value())
        
        self.pred.changePredictTextValue.connect(self.setPredictText)
        
        self.pred.start()

    def setProgressMax(self, val):
        self.extractProgressBar.setMaximum(val)    
     
    def setProgressVal(self, val):
        self.extractProgressBar.setValue(val)
        
    def setExtractText(self, text):
        self.extractTextBrowser.append(text)
        
    def setPredictText(self, text):
        self.trainTextBrowser.append(text)
        
    def saveWikipedia(self, data):
         import pandas as pd
         
         df = pd.DataFrame (np.array(data))
     
         filepath,_ = QtWidgets.QFileDialog.getSaveFileName(None,"QFileDialog.getSaveFileName()", "","All Files (*);;Excel (*.xlsx)")
    
         df.to_excel(filepath, index=False)
        
    def loadOnPush(self):
        self.fileNamePred, _ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        self.trainButton.setDisabled(False)
        
       

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
app = QtWidgets.QApplication(sys.argv)
application = ApplicationWindow()
application.show()
sys.exit(app.exec_())