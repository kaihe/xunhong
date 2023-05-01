import sys
import os
from PyQt5 import QtGui,QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QTextEdit, QLineEdit, QLabel, QScrollArea,QWidget,QStackedLayout, QListWidget, QComboBox, QListWidgetItem
from PyQt5.QtCore import Qt
import json
import copy

ANNO_KEYS = ['Thought: ','Action: ','ActionInput: ','MyAnswer: ']

class window(QDialog):
    def __init__(self):
        super(window,self).__init__()

        self.title = 'bot chat'
        self.left = 50
        self.top = 50
        self.width = 1800
        self.height = 900

        self.conv_id = 0
        self.line_id = 0

        self.src_file = r'raw_data\metra_langchain_dialog.json'
        self.anno_save_file = r'raw_data\_anno' + os.path.basename(self.src_file)

        self.initWindow()

    def initWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)
        self.load_convs()
        self.main_layout = QHBoxLayout()

        self.conv_widget = QListWidget()
        self._init_conv_area()

        self.anno_widget = QListWidget()
        self._init_anno_area()

        self.setLayout(self.main_layout)

    def load_convs(self):
        self.convs = json.load(open(self.src_file))
        if os.path.exists(self.anno_save_file) and False:
            self.conv_annos = json.load(open(self.anno_save_file))
        else:
            self.conv_annos = []
            for conv in self.convs:
                _conv_anno = []
                for line in conv:
                    if '悠悠：' in line:
                        _anno = copy.deepcopy(ANNO_KEYS)
                        _anno[-1] += line.replace('悠悠：','')
                        _conv_anno.append(_anno)
                    else:
                        _conv_anno.append([])
                self.conv_annos.append(_conv_anno)

        
    def _load_conv_widget(self):
        self.current_conv = self.convs[self.conv_id]
        self.conv_widget.clear()
        self.conv_widget.addItems(self.current_conv)

        for i, line in enumerate(self.current_conv):
            if '悠悠：' in line:
                item = self.conv_widget.item(i)
                item.setForeground(Qt.blue)

    def _init_conv_area(self):
        self.conv_widget.setFixedWidth(self.width//2)
        self.conv_widget.setWordWrap(True) 
        self.conv_widget.currentRowChanged.connect(self.switch_conv_line_event)
        self._load_conv_widget()
        self.main_layout.addWidget(self.conv_widget)

    def _save_anno_widget(self):
        self.conv_annos[self.conv_id][self.line_id] = [self.anno_widget.item(i).text() for i in range(self.anno_widget.count())]
        self._dump_anno_result()

    def _load_anno_widget(self, idx=0):
        self.current_anno = self.conv_annos[self.conv_id]
        self.line_id = idx
        self.anno_widget.clear()
        self.anno_widget.addItems(self.current_anno[idx])
    
        for i in range(self.anno_widget.count()):
            item = self.anno_widget.item(i)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def _init_anno_area(self):
        anno_area_layout = QGridLayout()

        # buttons
        button_next = QPushButton('Next')
        button_next.clicked.connect(self._next_conv)

        button_prev = QPushButton('Prev')
        button_prev.clicked.connect(self._prev_conv)
        
        button_add_thought = QPushButton('AddThought')
        
        button_save = QPushButton('Save')
        button_save.clicked.connect(self._dump_anno_result)

        anno_area_layout.addWidget(button_next,0,0)
        anno_area_layout.addWidget(button_prev,0,1)
        anno_area_layout.addWidget(button_add_thought, 0, 2)
        anno_area_layout.addWidget(button_save,0,3)

        # anno area
        self.anno_widget = QListWidget()
        # anno_group_box.setMaximumWidth(int(self.width/2))
        self._load_anno_widget()

        anno_area_layout.addWidget(self.anno_widget, 1,0,1,4)
        _widget = QWidget()
        _widget.setLayout(anno_area_layout)
        self.main_layout.addWidget(_widget)
    
    def switch_conv_line_event(self, index):
        self._save_anno_widget()
        self._load_anno_widget(idx=index)

    def _dump_anno_result(self):
        with open(self.anno_save_file, 'w+') as fout:
            json.dump(self.conv_annos, fout, ensure_ascii=False)

    def _next_conv(self):
        self._save_anno_widget()
        self.conv_id += 1
        self.conv_id %= len(self.convs)
        self._refresh_conv()
    
    def _prev_conv(self):
        self._save_anno_widget()
        self.conv_id -= 1
        self.conv_id %= len(self.convs)
        self._refresh_conv()

    def _refresh_conv(self):
        # update conv and anno
        self._load_conv_widget() 
        self._load_anno_widget(idx=1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = window()
    window.show()
    app.exit(app.exec_())
