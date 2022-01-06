from PyQt5.QtWidgets import QMainWindow, QApplication,QTableWidgetItem
from PyQt5.uic import loadUi
import numpy as np
import tools

class MainWindowWidget(QMainWindow):
    
    def __init__(self):

        QMainWindow.__init__(self)
        loadUi("assets/PD_UI.ui",self)
        self.setWindowTitle("PD")
        self.result = None

        self.tableResult.setColumnWidth(0, 202)
        self.tableResult.setColumnWidth(1, 202)
        self.tableResult.setColumnWidth(2, 202)
        self.tableResult.setColumnWidth(3, 202)

        self.GD.clicked.connect(self.GradientDescent)
        self.CG.clicked.connect(self.ConjugateGradient)
        self.N.clicked.connect(self.Newton)
        self.RP.clicked.connect(self.RandomPoint)

    def loadTable(self):
        self.tableResult.setRowCount(len(self.result))
        for count, i in enumerate(self.result):
            self.tableResult.setItem(count, 0, QTableWidgetItem(str(self.result[i][0][0])))
            self.tableResult.setItem(count, 1, QTableWidgetItem(str(self.result[i][0][1])))
            self.tableResult.setItem(count, 2, QTableWidgetItem(str(self.result[i][0][2])))
            self.tableResult.setItem(count, 3, QTableWidgetItem(str(self.result[i][1])))

    def GradientDescent(self):
        x0 = np.array([float(self.X.text()), float(self.Y.text()), float(self.Z.text())])
        self.result = tools.GradientDescentD3(x0, float(self.e.text()), int(self.iter.text()))
        self.loadTable()

    def ConjugateGradient(self):
        x0 = np.array([float(self.X.text()), float(self.Y.text()), float(self.Z.text())])
        self.result = tools.Conjugate_Gradient(x0, float(self.e.text()), int(self.iter.text()))
        self.loadTable()

    def Newton(self):
        x0 = np.array([float(self.X.text()), float(self.Y.text()), float(self.Z.text())])
        self.result = tools.Newton(x0, float(self.e.text()), int(self.iter.text()))
        self.loadTable()

    def RandomPoint(self):
        x = np.random.rand(3)*np.random.randint(1, 155)
        self.X.setText(str(x.item(0)))
        self.Y.setText(str(x.item(1)))
        self.Z.setText(str(x.item(2)))


app = QApplication([])
window = MainWindowWidget()
window.show()
app.exec_()


