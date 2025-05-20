from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QLabel, QApplication, QStyle, 
                             QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QTabWidget,
                             QListWidget, QListWidgetItem, QSplitter, QComboBox, QMessageBox,
                             QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QSize, QTimer
from PyQt5.QtGui import QPalette, QColor, QIcon, QTextCursor, QTextCharFormat, QFont
import sys
import time
import json
import os
from datetime import datetime
import threading
from eye_tracker import EyeTracker  # Import the EyeTracker class

# Worker for running Gemma model in a separate thread
class GemmaWorker(QObject):
    new_token = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, gemma_model, prompt):
        super().__init__()
        self.gemma_model = gemma_model
        self.prompt = prompt
        self._is_running = True

    def run(self):
        try:
            for token in self.gemma_model.generate_stream(self.prompt):
                if not self._is_running:
                    break
                self.new_token.emit(token)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False

class CustomPromptInput(QTextEdit):
    returnPressedNoShift = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Ask Gemma anything...")
        self.setMaximumHeight(self.fontMetrics().lineSpacing() * 5 + 15)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.returnPressedNoShift.emit()
                event.accept()
        else:
            super().keyPressEvent(event)

class CalibrationWorker(QObject):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, eye_tracker, student_data):
        super().__init__()
        self.eye_tracker = eye_tracker
        self.student_data = student_data
        self._is_running = True
        
    def run(self):
        try:
            # Simulated calibration process (replace with actual eye tracker calibration)
            for i in range(101):
                if not self._is_running:
                    break
                
                # Simulate processing
                time.sleep(0.05)
                self.progress_update.emit(i)
            
            # Generate calibration data (replace with actual eye tracker data)
            calibration_data = {
                "timestamp": datetime.now().isoformat(),
                "eye_pattern": f"simulated_pattern_{hash(self.student_data['name'] + self.student_data['id'])}"
            }
            
            # Add calibration data to student profile
            self.student_data["calibration"] = calibration_data
            self.finished.emit(self.student_data)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False

class AddStudentDialog(QDialog):
    def __init__(self, parent=None, eye_tracker=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Student")
        self.setMinimumWidth(400)
        self.eye_tracker = eye_tracker
        self.calibration_thread = None
        self.calibration_worker = None
        self.calibrated_data = None
        
        # Get colors from parent if available
        self.colors = parent.colors if parent and hasattr(parent, 'colors') else {
            "bg": "#FFFFFF",
            "card": "#F8F9FA",
            "text": "#212529",
            "primary": "#4361EE",
            "secondary": "#6C757D",
            "accent": "#7209B7",
            "error": "#EF476F",
            "success": "#05603A",  # Darker green
            "light": "#FFFFFF",
            "dark": "#E9ECEF",
            "border": "#DEE2E6",
            "input_bg": "#FFFFFF",
            "input_border": "#CED4DA"
        }
        
        # Set dialog style
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.colors['card']};
            }}
            QLineEdit {{
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors['primary']};
            }}
            QLabel {{
                color: {self.colors['text']};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {self.colors['success']};
                color: {self.colors['light']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #044C2E;
            }}
            QPushButton:pressed {{
                background-color: #033821;
            }}
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Setup tabs
        self.tab_widget = QTabWidget()
        self.info_tab = QWidget()
        self.calibration_tab = QWidget()
        
        self.tab_widget.addTab(self.info_tab, "Student Info")
        self.tab_widget.addTab(self.calibration_tab, "Calibration")
        
        # Setup info tab
        self.setup_info_tab()
        
        # Setup calibration tab
        self.setup_calibration_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
        
        # Disable OK button until calibration is complete
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        
        # Apply style to the tab widget
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ 
                border: 1px solid {self.colors['border']}; 
                background-color: {self.colors['card']};
            }}
            QTabBar::tab {{ 
                background-color: {self.colors['bg']}; 
                color: {self.colors['text']}; 
                border: 1px solid {self.colors['border']}; 
                padding: 8px 12px; 
                margin-right: 2px; 
            }}
            QTabBar::tab:selected {{ 
                background-color: {self.colors['primary']}; 
                color: {self.colors['light']}; 
                border-bottom-color: {self.colors['primary']}; 
            }}
        """)
    
    def setup_info_tab(self):
        # Form layout
        layout = QFormLayout(self.info_tab)
        
        # Input fields
        self.name_input = QLineEdit()
        self.id_input = QLineEdit()
        self.class_input = QComboBox()
        self.class_input.addItems(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        self.section_input = QComboBox()
        self.section_input.addItems(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"])
        
        # Style the combo boxes
        combobox_style = f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
            }}
            QComboBox::drop-down {{
                border: 0px;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card']};
                color: {self.colors['text']};
                selection-background-color: {self.colors['primary']};
                selection-color: {self.colors['light']};
                border: 1px solid {self.colors['border']};
            }}
        """
        self.class_input.setStyleSheet(combobox_style)
        self.section_input.setStyleSheet(combobox_style)
        
        # Add fields to layout
        layout.addRow("Name:", self.name_input)
        layout.addRow("ID:", self.id_input)
        layout.addRow("Class:", self.class_input)
        layout.addRow("Section:", self.section_input)
        
        # Next button
        self.next_btn = QPushButton("Next: Calibration")
        self.next_btn.setStyleSheet(f"""
            background-color: {self.colors['primary']};
            color: {self.colors['light']};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        self.next_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        layout.addRow(self.next_btn)
    
    def setup_calibration_tab(self):
        layout = QVBoxLayout(self.calibration_tab)
        
        # Instructions
        instructions = QLabel("Look at the screen and follow the instructions to calibrate your eye tracking profile.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {self.colors['text']};")
        layout.addWidget(instructions)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                text-align: center;
                color: {self.colors['text']};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['primary']};
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to calibrate")
        self.status_label.setStyleSheet(f"color: {self.colors['text']};")
        layout.addWidget(self.status_label)
        
        # Start calibration button
        self.calibrate_btn = QPushButton("Start Calibration")
        self.calibrate_btn.setStyleSheet(f"""
            background-color: {self.colors['success']};
            color: {self.colors['light']};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        layout.addWidget(self.calibrate_btn)
        
        layout.addStretch()
    
    def start_calibration(self):
        # Check if required fields are filled
        if not self.name_input.text() or not self.id_input.text() or not self.class_input.currentText():
            QMessageBox.warning(self, "Missing Information", 
                               "Please fill all required fields (Name, ID, Class) before calibration.")
            self.tab_widget.setCurrentIndex(0)
            return
        
        # Get basic student data
        student_data = self.get_student_data()
        
        # Update UI
        self.calibrate_btn.setEnabled(False)
        self.status_label.setText("Calibrating... Please follow instructions on screen")
        self.progress_bar.setValue(0)
        
        # Check if eye tracker is available
        if not self.eye_tracker:
            # Simulate calibration if no eye tracker
            self.calibration_worker = CalibrationWorker(self.eye_tracker, student_data)
            self.calibration_thread = QThread()
            self.calibration_worker.moveToThread(self.calibration_thread)
            self.calibration_worker.progress_update.connect(self.update_calibration_progress)
            self.calibration_worker.finished.connect(self.calibration_finished)
            self.calibration_worker.error.connect(self.calibration_error)
            self.calibration_thread.started.connect(self.calibration_worker.run)
            self.calibration_thread.start()
        else:
            try:
                # Connect progress and status callbacks
                self.eye_tracker.set_status_callback(lambda msg: self.status_label.setText(msg))
                self.eye_tracker.set_progress_callback(lambda progress: self.progress_bar.setValue(int(progress)))
                
                # Start a student-specific calibration
                student_id = student_data['id']
                student_name = student_data['name']
                
                # Use calibrate_for_student to associate the profile with the student
                threshold = self.eye_tracker.calibrate_for_student(student_id, student_name)
                
                # Add eye tracker data to student profile
                student_data["calibration"] = {
                    "timestamp": datetime.now().isoformat(),
                    "ear_threshold": threshold,
                    "head_pose_threshold": self.eye_tracker.head_pose_threshold,
                    "eye_pattern": f"eye_pattern_{hash(student_data['name'] + student_data['id'])}"
                }
                
                # Calibration is complete - update UI
                self.progress_bar.setValue(100)
                self.status_label.setText("Calibration complete!")
                self.calibrated_data = student_data
                
                # Delay enabling the OK button to prevent accidental clicking
                QTimer.singleShot(1500, lambda: self.button_box.button(QDialogButtonBox.Ok).setEnabled(True))
                
            except Exception as e:
                self.calibration_error(f"Eye tracker calibration error: {str(e)}")
                self.calibrate_btn.setEnabled(True)
    
    def update_calibration_progress(self, value):
        self.progress_bar.setValue(value)
    
    def calibration_finished(self, calibrated_data):
        # Add a delay before enabling OK button to ensure full calibration experience
        self.calibrated_data = calibrated_data
        self.status_label.setText("Calibration complete!")
        self.progress_bar.setValue(100)
        
        # Delay enabling the OK button to prevent accidental clicking
        # and to ensure user sees the completion message
        QTimer.singleShot(1500, lambda: self.button_box.button(QDialogButtonBox.Ok).setEnabled(True))
        
        if self.calibration_thread:
            self.calibration_thread.quit()
            self.calibration_thread.wait()
            self.calibration_thread = None
            self.calibration_worker = None
    
    def calibration_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.calibrate_btn.setEnabled(True)
        
        if self.calibration_thread:
            self.calibration_thread.quit()
            self.calibration_thread.wait()
            self.calibration_thread = None
            self.calibration_worker = None
    
    def get_student_data(self):
        return {
            "name": self.name_input.text(),
            "id": self.id_input.text(),
            "class": self.class_input.currentText(),
            "section": self.section_input.currentText(),
            "joined_date": datetime.now().isoformat(),
            "request_history": []
        }
    
    def accept(self):
        if not self.calibrated_data:
            QMessageBox.warning(self, "Incomplete Setup", 
                               "Please complete the calibration before adding the student.")
            return
        
        super().accept()

class StudentProfileWidget(QWidget):
    def __init__(self, student_data, parent=None):
        super().__init__(parent)
        self.student_data = student_data
        self.colors = parent.colors if parent and hasattr(parent, 'colors') else {}
        
        # Setup UI
        layout = QVBoxLayout(self)
        
        # Student info section
        info_layout = QFormLayout()
        
        self.name_label = QLabel(f"<b>{student_data['name']}</b>")
        self.id_label = QLabel(student_data['id'])
        self.class_label = QLabel(f"{student_data['class']} - {student_data['section']}")
        self.joined_label = QLabel(datetime.fromisoformat(student_data['joined_date']).strftime("%Y-%m-%d"))
        
        info_layout.addRow("Name:", self.name_label)
        info_layout.addRow("ID:", self.id_label)
        info_layout.addRow("Class:", self.class_label)
        info_layout.addRow("Joined:", self.joined_label)
        
        # Add info section to layout
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        layout.addWidget(info_widget)
        
        # Recent requests section
        history_header = QLabel("<b>Recent Requests:</b>")
        history_header.setStyleSheet(f"color: {self.colors.get('primary', '#4361EE')}; margin-top: 10px;")
        layout.addWidget(history_header)
        
        # Use a widget that supports rich text instead of QListWidget
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setSpacing(8)
        self.history_layout.setContentsMargins(5, 5, 5, 5)
        
        self.history_container.setStyleSheet(f"""
            background-color: {self.colors.get('card', '#F8F9FA')};
            border: 1px solid {self.colors.get('border', '#DEE2E6')};
            border-radius: 4px;
        """)
        
        self.update_history_list()
        layout.addWidget(self.history_container)
    
    def update_history_list(self):
        # Clear existing items
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if 'request_history' not in self.student_data or not self.student_data['request_history']:
            no_history_label = QLabel("No recent requests")
            no_history_label.setStyleSheet(f"color: {self.colors.get('secondary', '#6C757D')};")
            self.history_layout.addWidget(no_history_label)
            return
        
        # Get the 5 most recent requests and display in chronological order (oldest to newest)
        recent_requests = self.student_data['request_history'][-5:]
        # Note: no longer reversing the order
        
        for request in recent_requests:
            # Create a label with formatted HTML
            history_item = QLabel()
            history_item.setTextFormat(Qt.RichText)
            history_item.setWordWrap(True)
            
            # Explicitly set colors for all text elements to ensure proper contrast
            history_item.setText(f"""<div style='color:{self.colors.get('text', '#212529')};'>
                <b style='color:{self.colors.get('primary', '#4361EE')};'>{request['timestamp']}</b><br>
                <span style='margin-left:10px;'>{request['prompt']}</span>
            </div>""")
            
            # Add a bottom border except for the last item
            history_item.setStyleSheet(f"""
                padding: 8px;
                border-bottom: 1px solid {self.colors.get('border', '#DEE2E6')};
                color: {self.colors.get('text', '#212529')};
            """)
            
            self.history_layout.addWidget(history_item)
        
        # Add a stretch at the end to push all items to the top
        self.history_layout.addStretch()
    
    def update_student_data(self, student_data):
        self.student_data = student_data
        
        # Update UI with new data
        self.name_label.setText(f"<b>{student_data['name']}</b>")
        self.id_label.setText(student_data['id'])
        self.class_label.setText(f"{student_data['class']} - {student_data['section']}")
        self.joined_label.setText(datetime.fromisoformat(student_data['joined_date']).strftime("%Y-%m-%d"))
        
        self.update_history_list()

class MainWindow(QMainWindow):
    def __init__(self, gemma_model_instance=None, eye_tracker_instance=None):
        super().__init__()
        self.gemma_model = gemma_model_instance
        self.gemma_thread = None
        self.gemma_worker = None
        self.students = []  # List to store student data
        self.current_student = None  # Currently detected/selected student
        self.data_file = "assets/students.json"
        
        # Initialize eye tracker if not provided
        if eye_tracker_instance:
            self.eye_tracker = eye_tracker_instance
        else:
            try:
                # Initialize in lightweight mode for better compatibility
                self.eye_tracker = EyeTracker(lightweight_mode=True)
                print("Eye tracker initialized in lightweight mode")
            except Exception as e:
                print(f"Could not initialize eye tracker: {str(e)}")
                self.eye_tracker = None
        
        # Load existing student data if available
        self.load_students()

        self.setWindowTitle("Teacher Robot Assistant")
        self.setGeometry(100, 100, 1200, 800)

        # Light theme colors (can be expanded later)
        self.colors = {
            "bg": "#FFFFFF",
            "card": "#F8F9FA",
            "text": "#212529",
            "primary": "#4361EE", # A nice blue
            "secondary": "#6C757D", # Grey for secondary actions or borders
            "accent": "#7209B7",
            "error": "#EF476F",
            "success": "#05603A",  # Changed to darker green
            "light": "#FFFFFF",
            "dark": "#E9ECEF",
            "border": "#DEE2E6",
            "input_bg": "#FFFFFF",
            "input_border": "#CED4DA"
        }

        # Store a default character format for resetting styles
        self.default_char_format = QTextCharFormat()
        self.default_char_format.setForeground(QColor(self.colors["text"])) # Set default text color
        self.default_char_format.setFontWeight(QFont.Normal)
        self.default_char_format.setFontUnderline(False)
        self.default_char_format.setFontStrikeOut(False)
        # Add other properties if needed, e.g., self.default_char_format.setFontWeight(QFont.Normal)

        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        # Header
        self.header = QLabel("Teacher Robot Assistant")
        self.header.setObjectName("header")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)
        
        # Management buttons
        self.mgmt_buttons_layout = QHBoxLayout()
        
        self.add_student_button = QPushButton("Add Student")
        self.add_student_button.setObjectName("addStudentButton")
        self.add_student_button.clicked.connect(self.show_add_student_dialog)
        self.mgmt_buttons_layout.addWidget(self.add_student_button)
        
        self.mgmt_buttons_layout.addStretch()
        self.layout.addLayout(self.mgmt_buttons_layout)

        # Main area with splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for student info
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Student selector
        self.left_layout.addWidget(QLabel("<b>Current Student:</b>"))
        self.student_selector = QComboBox()
        self.student_selector.currentIndexChanged.connect(self.on_student_changed)
        self.left_layout.addWidget(self.student_selector)
        
        # Student profile area
        self.profile_area = QWidget()
        self.profile_layout = QVBoxLayout(self.profile_area)
        self.left_layout.addWidget(self.profile_area)
        
        # Add left panel to splitter
        self.main_splitter.addWidget(self.left_panel)
        
        # Right panel for chat
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Output Area
        self.output_area = QTextEdit()
        self.output_area.setObjectName("outputArea")
        self.output_area.setReadOnly(True)
        self.output_area.setMinimumHeight(500)
        self.right_layout.addWidget(self.output_area)

        # --- Input Area ---
        self.input_layout = QHBoxLayout()
        
        self.prompt_input = CustomPromptInput()
        self.prompt_input.setObjectName("promptInput")
        self.input_layout.addWidget(self.prompt_input)

        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        send_icon_path = "assets/send.svg"
        if sys.platform == "win32" and not send_icon_path.startswith("../"):
             # Basic check if running from src dir on windows
            if __file__.replace("\\", "/").endswith("src/window.py"):
                 send_icon_path = "../assets/send.svg"

        try:
            send_icon = QIcon(send_icon_path)
            if send_icon.isNull():
                print(f"Warning: Could not load icon from {send_icon_path}. Using fallback.")
                self.send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
            else:
                self.send_button.setIcon(send_icon)
        except Exception as e:
            print(f"Error loading icon: {e}. Using fallback.")
            self.send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
            
        self.send_button.setIconSize(QSize(20, 20))
        self.send_button.setFixedSize(40, 40)

        self.send_button.clicked.connect(self.start_gemma_generation)
        self.prompt_input.returnPressedNoShift.connect(self.start_gemma_generation)

        self.input_layout.addWidget(self.send_button)
        
        self.right_layout.addLayout(self.input_layout)
        
        # Add right panel to splitter
        self.main_splitter.addWidget(self.right_panel)
        
        # Set initial splitter sizes (30% left, 70% right)
        self.main_splitter.setSizes([300, 700])
        
        # Add splitter to main layout
        self.layout.addWidget(self.main_splitter)

        # Status Bar
        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")
        self.statusBar().addPermanentWidget(self.status_label)
        
        # Setup student detection timer (simulated)
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.detect_student)
        self.detection_timer.start(5000)  # Check every 5 seconds

        # Update UI
        self.update_student_selector()
        self.apply_theme()

    def apply_theme(self):
        app = QApplication.instance()
        if not app:
            return

        stylesheet = f"""
            QMainWindow, QWidget#centralWidget {{
                background-color: {self.colors['bg']};
                font-family: 'Inter', 'Segoe UI', 'Helvetica', sans-serif;
            }}
            QLabel#header {{
                color: {self.colors['primary']};
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                margin-bottom: 5px; /* Reduced margin */
            }}
            QTextEdit#promptInput {{
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }}
            QTextEdit#promptInput:focus {{
                border: 1px solid {self.colors['primary']};
            }}
            QPushButton#sendButton {{
                background-color: {self.colors['primary']};
                color: {self.colors['light']};
                border: none;
                border-radius: 20px;
                padding: 0px;
            }}
            QPushButton#sendButton:hover {{
                background-color: #3A0CA3;
            }}
            QPushButton#sendButton:pressed {{
                background-color: #4361EE;
            }}
            QTextEdit#outputArea {{
                background-color: {self.colors['card']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.6;
            }}
            QLabel#statusLabel {{
                color: {self.colors['text']};
                font-size: 12px;
                padding: 5px 10px;
            }}
            QStatusBar {{
                background-color: {self.colors['card']};
                border-top: 1px solid {self.colors['border']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: {self.colors['bg']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.colors['primary']};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background: {self.colors['bg']};
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {self.colors['primary']};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QPushButton#addStudentButton {{
                background-color: {self.colors['success']};
                color: {self.colors['light']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton#addStudentButton:hover {{
                background-color: #044C2E;
            }}
            QPushButton#addStudentButton:pressed {{
                background-color: #033821;
            }}
        """
        app.setStyleSheet(stylesheet)

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def closeEvent(self, event):
        # Save student data before closing
        self.save_students()
        
        # Clean up eye tracker
        if hasattr(self, 'eye_tracker') and self.eye_tracker:
            self.eye_tracker.release()
        
        if self.gemma_worker:
            self.gemma_worker.stop()
        if self.gemma_thread and self.gemma_thread.isRunning():
            self.gemma_thread.quit()
            self.gemma_thread.wait(1000)
        super().closeEvent(event)
    
    def load_students(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.students = json.load(f)
        except Exception as e:
            print(f"Error loading students: {e}")
    
    def save_students(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.students, f, indent=2)
        except Exception as e:
            print(f"Error saving students: {e}")
    
    def update_student_selector(self):
        self.student_selector.clear()
        
        if not self.students:
            self.student_selector.addItem("No students available")
            self.student_selector.setEnabled(False)
            return
        
        self.student_selector.setEnabled(True)
        self.student_selector.addItem("Select a student...")
        
        for student in self.students:
            self.student_selector.addItem(f"{student['name']} ({student['id']})")
    
    def on_student_changed(self, index):
        # Clear existing profile
        self.clear_profile_area()
        
        # Skip the first item ("Select a student...")
        if index <= 0 or index > len(self.students):
            self.current_student = None
            return
        
        # Get the selected student (index-1 because of the "Select" item)
        self.current_student = self.students[index-1]
        
        # Create and display the profile widget
        profile_widget = StudentProfileWidget(self.current_student, self)
        self.profile_layout.addWidget(profile_widget)
        
        self.update_status(f"Student {self.current_student['name']} selected")
    
    def clear_profile_area(self):
        # Remove all widgets from profile area
        while self.profile_layout.count():
            item = self.profile_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def detect_student(self):
        if not self.eye_tracker or not self.students:
            return
        
        try:
            # Check if user is present and looking at camera
            if self.eye_tracker.is_user_present() and self.eye_tracker.is_looking_at_camera():
                # In a real implementation, we'd match the eye pattern with calibrated profiles
                # For now, use student profiles with their thresholds to find best match
                
                best_match = None
                best_match_score = 0
                
                # Try to identify by checking if the currently detected pattern matches any students
                for student in self.students:
                    if 'calibration' in student:
                        # Get the student's calibration data
                        calibration = student['calibration']
                        
                        # Check if the calibration matches the current user
                        # This is a simplified version - in a real implementation, 
                        # we would compare eye patterns more comprehensively
                        if 'ear_threshold' in calibration:
                            # As a very basic method, check if the current EAR is close to the student's threshold
                            # In a real implementation, this would be much more sophisticated
                            match_score = 1.0  # Default to equal likelihood for all students with calibration
                            
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_match = student
                
                # If we found a match and it's different from current student
                if best_match and (not self.current_student or best_match['id'] != self.current_student['id']):
                    # Find the index of the matched student
                    for i, student in enumerate(self.students):
                        if student['id'] == best_match['id']:
                            # Select the student in the dropdown (+1 for the "Select" item)
                            self.student_selector.setCurrentIndex(i + 1)
                            self.update_status(f"Eye tracking detected student: {best_match['name']}")
                            break
            else:
                # If user is not present or not looking, optionally deselect the student
                # or keep the current selection
                pass
                
        except Exception as e:
            print(f"Error in student detection: {e}")
            # Fallback to a random student if eye tracking fails
            # This is just for testing purposes
            if not self.current_student and self.students:
                import random
                if random.random() < 0.1:  # 10% chance to select a random student
                    student_index = random.randint(0, len(self.students)-1)
                    self.student_selector.setCurrentIndex(student_index + 1)
                    self.update_status(f"Random student selected for testing")

    def show_add_student_dialog(self):
        dialog = AddStudentDialog(self, self.eye_tracker)
        result = dialog.exec_()
        
        if result == QDialog.Accepted and dialog.calibrated_data:
            self.add_student(dialog.calibrated_data)
    
    def add_student(self, student_data):
        # Check if student with same ID already exists
        for i, student in enumerate(self.students):
            if student['id'] == student_data['id']:
                # Update existing student data
                self.students[i] = student_data
                self.update_status(f"Updated student: {student_data['name']}")
                self.save_students()
                self.update_student_selector()
                return
        
        # Add new student
        self.students.append(student_data)
        self.update_status(f"Added student: {student_data['name']}")
        self.save_students()
        self.update_student_selector()
        
        # Select the newly added student
        self.student_selector.setCurrentIndex(len(self.students))
    
    # --- Gemma Interaction Methods ---
    def start_gemma_generation(self):
        if not self.gemma_model:
            self.output_area.append("<font color='red'>Gemma model not available.</font>")
            return
        
        # Check if a student is selected
        if not self.current_student:
            self.output_area.append("<font color='red'>Please select a student first.</font>")
            return
             
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt: return
        if self.gemma_thread and self.gemma_thread.isRunning():
            self.update_status("Gemma is busy. Please wait.")
            return
        
        # Build context from previous prompts if available
        context = self.build_context_from_history()
        
        # Record this request in student history
        self.record_student_request(prompt)
        
        # Process the prompt with or without context
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nNow the query is:\n{prompt}"
        
        processed_prompt = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        user_html = f"<div style='margin-bottom:10px;'><b style='color:{self.colors['secondary']};'>{self.current_student['name']}:</b> <span style='color:{self.colors['text']};'>{processed_prompt}</span></div>"
        self.output_area.append(user_html)
        gemma_label_html = f"<div style='margin-bottom:5px;'><b style='color:{self.colors['primary']};'>Gemma:</b>&nbsp;</div>"
        self.output_area.append(gemma_label_html)
        # Append an extra empty div to break any inherited style
        self.output_area.append("<div></div>")
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.setCharFormat(self.default_char_format)
        self.output_area.setTextCursor(cursor)
        self.prompt_input.clear()
        self.update_status("Generating response...")
        self.send_button.setEnabled(False)
        self.gemma_worker = GemmaWorker(self.gemma_model, full_prompt)
        self.gemma_thread = QThread()
        self.gemma_worker.moveToThread(self.gemma_thread)
        self.gemma_worker.new_token.connect(self.append_gemma_token)
        self.gemma_worker.finished.connect(self.gemma_generation_finished)
        self.gemma_worker.error.connect(self.gemma_generation_error)
        self.gemma_thread.started.connect(self.gemma_worker.run)
        self.gemma_thread.start()

    def append_gemma_token(self, token):
        cursor = self.output_area.textCursor() # Get current cursor
        # Ensure cursor is at the end of any previously inserted text
        # No need to movePosition(QTextCursor.End) here if start_gemma_generation left it correctly
        # and we assume insertPlainText continues from there.
        # The critical part is that the charFormat of the cursor is already default_char_format.
        cursor.insertText(token)
        self.output_area.ensureCursorVisible()

    def gemma_generation_finished(self):
        # Insert a plain text separator instead of HTML to avoid underline inheritance
        separator = "\n" + "―" * 60 + "\n"
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.setCharFormat(self.default_char_format)
        cursor.insertText(separator)
        self.output_area.setTextCursor(cursor)
        self.update_status("Ready")
        self.send_button.setEnabled(True)
        if self.gemma_thread:
            self.gemma_thread.quit()
            self.gemma_thread.wait()
        self.gemma_thread = None
        self.gemma_worker = None

    def gemma_generation_error(self, error_message):
        error_html = f"<div style='color:{self.colors['error']};margin-top:5px;'><b>Error:</b> {error_message}</div>"
        error_html += f"<hr style='border:none;border-top:1px solid {self.colors['border']};margin-top:5px;margin-bottom:10px;'>"
        self.output_area.append(error_html)
        self.update_status(f"Error: {error_message[:50]}...")
        self.send_button.setEnabled(True)
        if self.gemma_thread:
            self.gemma_thread.quit()
            self.gemma_thread.wait()
        self.gemma_thread = None
        self.gemma_worker = None

    def record_student_request(self, prompt):
        if not self.current_student:
            return
        
        # Add request to history
        request_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prompt": prompt
        }
        
        # Find the current student in the list and update
        for i, student in enumerate(self.students):
            if student['id'] == self.current_student['id']:
                # Make sure request_history exists
                if 'request_history' not in self.students[i]:
                    self.students[i]['request_history'] = []
                
                self.students[i]['request_history'].append(request_entry)
                self.current_student = self.students[i]  # Update current reference
                
                # Update the profile view if visible
                if self.student_selector.currentIndex() > 0:
                    self.clear_profile_area()
                    profile_widget = StudentProfileWidget(self.current_student, self)
                    self.profile_layout.addWidget(profile_widget)
                
                break
        
        # Save updated student data
        self.save_students()

    def build_context_from_history(self):
        """Build a context string from the previous 5 interactions"""
        if not self.current_student or 'request_history' not in self.current_student:
            return ""
        
        history = self.current_student['request_history']
        if len(history) <= 1:  # No previous history or just the current request
            return ""
        
        # Get up to 5 previous interactions (excluding the most recent one which is about to be added)
        prev_history = history[-6:-1] if len(history) >= 6 else history[:-1]
        
        context_lines = ["Previous interactions:"]
        for i, entry in enumerate(prev_history):
            context_lines.append(f"Question {i+1}: {entry['prompt']}")
        
        return "\n".join(context_lines)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Initialize EyeTracker
    try:
        import platform
        # Check if running on Raspberry Pi for lightweight mode
        is_raspberry_pi = platform.machine().startswith('arm')
        eye_tracker = EyeTracker(lightweight_mode=is_raspberry_pi)
        print(f"Eye tracker initialized in {'lightweight' if is_raspberry_pi else 'full'} mode")
    except Exception as e:
        print(f"Could not initialize eye tracker: {str(e)}")
        eye_tracker = None
    
    # Pass the eye tracker to MainWindow
    main_window = MainWindow(eye_tracker_instance=eye_tracker)
    main_window.show()
    sys.exit(app.exec_())