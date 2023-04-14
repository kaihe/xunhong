import sys

from PyQt5 import QtGui,QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QTextEdit, QLineEdit

from PyQt5.QtGui import QPixmap
from metra_conv_bot import get_bot

class window(QDialog):
    def __init__(self):
        super(window,self).__init__()

        self.title = 'bot chat'
        self.left = 50
        self.top = 50
        self.width = 1100
        self.height = 530
        self.initWindow()

    def initWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)

        self.vbox = QVBoxLayout()
        # self.add_buttons()
        self.add_conv_area()
        self.setLayout(self.vbox)
        self.bot = get_bot()
        self.history = []
        
    def add_conv_area(self):
        conv_group_box = QGroupBox()
        self.output_te = QTextEdit(readOnly=True)
        self.input_le = QLineEdit(returnPressed=self.on_return_pressed)
        
        conv_layout = QtWidgets.QVBoxLayout()
        conv_layout.addWidget(self.output_te)
        conv_layout.addWidget(self.input_le)

        conv_group_box.setLayout(conv_layout)
        self.vbox.addWidget(conv_group_box)

    def update_chat(self):
        round_text = '\n'.join(self.history)
        self.output_te.setPlainText(round_text)

    @QtCore.pyqtSlot()
    def on_return_pressed(self):
        incoming_msg = self.input_le.text()
        if incoming_msg:
            bot_response = self.bot.run(incoming_msg)
            self.history.append(f'用户：{incoming_msg}')
            self.history.append(f'悠悠：{bot_response}')

            self.update_chat()
            self.input_le.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = window()
    window.show()
    app.exit(app.exec_())
