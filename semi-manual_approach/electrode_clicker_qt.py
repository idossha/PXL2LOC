#!/usr/bin/env python3
"""
GSN-256 Electrode Coordinate Clicker - PyQt5 Version
Click on electrode positions to record their pixel coordinates.
"""

import sys
import os
import csv
import math
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np


class ElectrodeClickerQt(QMainWindow):
    def __init__(self, image_path=None):
        super().__init__()
        self.image_path = image_path
        self.original_image = None
        
        # Initialize variables
        self.electrodes = []
        self.current_electrode_num = 1
        self.selected_electrode_idx = None
        self.circle_radius = 8
        self.default_color = "red"
        self.auto_save_interval = 5
        self.scale_factor = 1.0
        self.move_step = 1
        self.insertion_mode = "automatic"  # "automatic" or "manual"
        self.save_filename = "electrode_coordinates.csv"
        
        # Color mapping
        self.color_map = {
            'red': (255, 0, 0),
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255)
        }
        
        self.setupUI()
        
        # Load image if provided
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
            self.load_data()
            
    def load_image(self, image_path):
        """Load an image file"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            QMessageBox.critical(self, "Error", f"Could not load image: {image_path}")
            return False
        
        # Convert BGR to RGB for Qt
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Reset electrodes for new image
        self.electrodes = []
        self.current_electrode_num = 1
        self.selected_electrode_idx = None
        
        # Update display
        self.update_display()
        self.update_status()
        
        # Set default save filename based on image name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.save_filename = f"{base_name}_coordinates.csv"
        self.filename_input.setText(self.save_filename)
        
        return True
        
    def select_image(self):
        """Open file dialog to select an image"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image File", 
            "", 
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if filename:
            if self.load_image(filename):
                self.current_image_label.setText(os.path.basename(filename))
                
    def update_save_filename(self, text):
        """Update the save filename"""
        self.save_filename = text if text else "electrode_coordinates.csv"
        
    def set_insertion_mode(self, mode):
        """Set the electrode insertion mode"""
        self.insertion_mode = mode
        self.update_status()
        
    def setupUI(self):
        """Setup the main UI with panels"""
        self.setWindowTitle("GSN-256 Electrode Clicker")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image display
        self.image_widget = ImageWidget(self)
        splitter.addWidget(self.image_widget)
        
        # Right panel - Controls
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Set initial splitter sizes (70% image, 30% controls)
        splitter.setSizes([980, 420])
        
    def create_control_panel(self):
        """Create the control panel with all controls and instructions"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("CONTROL PANEL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # File Selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Image selection button
        select_image_btn = QPushButton("Select Image (.png/.jpg)")
        select_image_btn.clicked.connect(self.select_image)
        file_layout.addWidget(select_image_btn)
        
        # Current image label
        self.current_image_label = QLabel("No image loaded")
        self.current_image_label.setWordWrap(True)
        self.current_image_label.setStyleSheet("color: gray; font-size: 11px;")
        file_layout.addWidget(self.current_image_label)
        
        # Save filename input
        filename_label = QLabel("Save filename:")
        file_layout.addWidget(filename_label)
        
        self.filename_input = QLineEdit(self.save_filename)
        self.filename_input.textChanged.connect(self.update_save_filename)
        file_layout.addWidget(self.filename_input)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Insertion Mode section
        mode_group = QGroupBox("Insertion Mode")
        mode_layout = QVBoxLayout()
        
        # Radio buttons for mode selection
        self.auto_mode_radio = QRadioButton("Automatic (E001 - E256)")
        self.auto_mode_radio.setChecked(True)
        self.auto_mode_radio.toggled.connect(lambda checked: self.set_insertion_mode("automatic" if checked else "manual"))
        mode_layout.addWidget(self.auto_mode_radio)
        
        self.manual_mode_radio = QRadioButton("Manual (Custom names)")
        mode_layout.addWidget(self.manual_mode_radio)
        
        mode_help = QLabel("Manual mode: Enter custom electrode names\n(e.g., AF1, F7, Cz)")
        mode_help.setStyleSheet("color: gray; font-size: 10px; margin-left: 20px;")
        mode_layout.addWidget(mode_help)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Status section
        status_group = QGroupBox("Status")
        status_layout = QGridLayout()
        
        self.status_labels = {
            'mode': QLabel(f"Mode: {self.insertion_mode.capitalize()}"),
            'next': QLabel(f"Next Electrode: E{self.current_electrode_num:03d}"),
            'progress': QLabel(f"Progress: {len(self.electrodes)}/256"),
            'remaining': QLabel(f"Remaining: {256 - len(self.electrodes)}"),
            'selected': QLabel("Selected: None"),
            'color': QLabel(f"Default Color: {self.default_color.upper()}")
        }
        
        row = 0
        for label in self.status_labels.values():
            status_layout.addWidget(label, row, 0)
            row += 1
            
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Circle Size Control
        size_group = QGroupBox("Circle Size")
        size_layout = QVBoxLayout()
        
        self.radius_label = QLabel(f"Radius: {self.circle_radius} pixels")
        size_layout.addWidget(self.radius_label)
        
        radius_slider = QSlider(Qt.Horizontal)
        radius_slider.setMinimum(3)
        radius_slider.setMaximum(20)
        radius_slider.setValue(self.circle_radius)
        radius_slider.valueChanged.connect(self.change_radius)
        size_layout.addWidget(radius_slider)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions = [
            "MOUSE CONTROLS:",
            "• Left-click: Place/Select electrode",
            "• Right-click: Remove electrode",
            "",
            "KEYBOARD CONTROLS:",
            "• Arrow Keys: Move selected electrode",
            "• Delete: Remove selected electrode",
            "• Escape: Clear selection",
            "• Ctrl+S: Save progress",
            "• Ctrl+Z: Undo last electrode"
        ]
        
        for instruction in instructions:
            label = QLabel(instruction)
            if instruction.endswith("CONTROLS:"):
                label.setStyleSheet("font-weight: bold; margin-top: 5px;")
            else:
                label.setStyleSheet("margin-left: 10px;")
            instructions_layout.addWidget(label)
            
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        # Action Buttons
        button_layout = QVBoxLayout()
        
        # Color selection
        color_button = QPushButton("Change Default Color")
        color_button.clicked.connect(self.change_default_color)
        button_layout.addWidget(color_button)
        
        # Reset button
        reset_button = QPushButton("Reset All Electrodes")
        reset_button.setStyleSheet("background-color: #ff6b6b; color: white;")
        reset_button.clicked.connect(self.reset_electrodes)
        button_layout.addWidget(reset_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push save button to bottom
        layout.addStretch()
        
        # Save and Quit button (at bottom)
        save_quit_button = QPushButton("SAVE && QUIT")
        save_quit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_quit_button.clicked.connect(self.save_and_quit)
        layout.addWidget(save_quit_button)
        
        # Auto-save info
        auto_save_label = QLabel(f"Auto-saves every {self.auto_save_interval} electrodes")
        auto_save_label.setAlignment(Qt.AlignCenter)
        auto_save_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(auto_save_label)
        
        return panel
        
    def change_radius(self, value):
        """Change circle radius from slider"""
        self.circle_radius = value
        self.radius_label.setText(f"Radius: {self.circle_radius} pixels")
        self.update_display()
        
    def change_default_color(self):
        """Change default electrode color"""
        colors = ["Red", "Black", "Blue", "Green", "Yellow", "Purple", "Cyan"]
        color, ok = QInputDialog.getItem(self, "Select Color", 
                                        "Choose default electrode color:", 
                                        colors, 0, False)
        if ok:
            self.default_color = color.lower()
            self.status_labels['color'].setText(f"Default Color: {self.default_color.upper()}")
            
    def reset_electrodes(self):
        """Reset all electrodes after confirmation"""
        reply = QMessageBox.question(self, 'Reset Electrodes', 
                                   'Are you sure you want to remove all electrodes?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.electrodes.clear()
            self.current_electrode_num = 1
            self.selected_electrode_idx = None
            self.update_display()
            self.update_status()
            
    def save_and_quit(self):
        """Save data and quit application"""
        self.save_data()
        QMessageBox.information(self, "Saved", "Electrode coordinates saved successfully!")
        self.close()
        
    def update_status(self):
        """Update all status labels"""
        self.status_labels['mode'].setText(f"Mode: {self.insertion_mode.capitalize()}")
        
        if self.insertion_mode == "automatic":
            self.status_labels['next'].setText(f"Next Electrode: E{self.current_electrode_num:03d}")
        else:
            self.status_labels['next'].setText("Next Electrode: Click to name")
            
        self.status_labels['progress'].setText(f"Progress: {len(self.electrodes)}/256")
        self.status_labels['remaining'].setText(f"Remaining: {256 - len(self.electrodes)}")
        
        if self.selected_electrode_idx is not None:
            electrode_name = self.electrodes[self.selected_electrode_idx]['electrode_name']
            self.status_labels['selected'].setText(f"Selected: {electrode_name}")
        else:
            self.status_labels['selected'].setText("Selected: None")
            
    def add_electrode(self, x, y):
        """Add a new electrode at the given position"""
        # Check if we already have 256 electrodes
        if len(self.electrodes) >= 256:
            QMessageBox.warning(self, "Limit Reached", "All 256 electrodes have been placed!")
            return
            
        if self.insertion_mode == "automatic":
            # Automatic naming: E001, E002, etc.
            electrode_name = f"E{self.current_electrode_num:03d}"
            electrode_number = self.current_electrode_num
            self.current_electrode_num += 1
        else:
            # Manual naming: Ask user for electrode name
            electrode_name, ok = QInputDialog.getText(
                self, 
                "Electrode Name", 
                "Enter electrode name (e.g., AF1, F7, Cz):",
                QLineEdit.Normal,
                ""
            )
            
            if not ok or not electrode_name:
                return  # User cancelled
                
            # Check for duplicate names
            if any(e['electrode_name'] == electrode_name for e in self.electrodes):
                QMessageBox.warning(self, "Duplicate Name", 
                                  f"Electrode '{electrode_name}' already exists!")
                return
                
            electrode_number = len(self.electrodes) + 1
            
        electrode_data = {
            'electrode_number': electrode_number,
            'electrode_name': electrode_name,
            'x': x,
            'y': y,
            'color': self.default_color
        }
        
        self.electrodes.append(electrode_data)
        self.selected_electrode_idx = None
        
        # Auto-save
        if len(self.electrodes) % self.auto_save_interval == 0:
            self.save_data(auto_save=True)
            
        self.update_display()
        self.update_status()
        
    def remove_electrode(self, idx):
        """Remove electrode at given index"""
        if 0 <= idx < len(self.electrodes):
            removed = self.electrodes.pop(idx)
            self.current_electrode_num = removed['electrode_number']
            if self.selected_electrode_idx == idx:
                self.selected_electrode_idx = None
            elif self.selected_electrode_idx is not None and self.selected_electrode_idx > idx:
                self.selected_electrode_idx -= 1
            self.update_display()
            self.update_status()
            
    def move_selected_electrode(self, dx, dy):
        """Move the selected electrode"""
        if self.selected_electrode_idx is not None:
            electrode = self.electrodes[self.selected_electrode_idx]
            new_x = electrode['x'] + dx
            new_y = electrode['y'] + dy
            
            # Keep within image bounds
            if 0 <= new_x < self.original_image.shape[1] and 0 <= new_y < self.original_image.shape[0]:
                electrode['x'] = new_x
                electrode['y'] = new_y
                self.update_display()
                
    def find_closest_electrode(self, x, y, max_distance=30):
        """Find the closest electrode to the given position"""
        if not self.electrodes:
            return None
            
        min_distance = float('inf')
        closest_idx = None
        
        for i, electrode in enumerate(self.electrodes):
            dx = electrode['x'] - x
            dy = electrode['y'] - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_idx = i
                
        return closest_idx
        
    def update_display(self):
        """Update the image display"""
        self.image_widget.update()
        
    def save_data(self, filename=None, auto_save=False):
        """Save electrode data to CSV"""
        if not self.electrodes:
            if not auto_save:
                QMessageBox.warning(self, "No Data", "No electrodes to save!")
            return
            
        # Use the filename from the input field
        if filename is None:
            filename = self.save_filename
            
        # Create DataFrame-like structure
        rows = []
        for electrode in self.electrodes:
            rows.append([
                electrode['electrode_name'],
                electrode['x'],
                electrode['y']
            ])
            
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['electrode_name', 'x', 'y'])
            writer.writerows(rows)
            
        if not auto_save:
            print(f"Saved {len(self.electrodes)} electrodes to {filename}")
        else:
            print(f"Auto-saved {len(self.electrodes)} electrodes")
            
    def load_data(self, filename=None):
        """Load existing electrode data"""
        if filename is None:
            filename = self.save_filename
            
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for i, row in enumerate(reader):
                        # Handle both old and new formats
                        if 'electrode_name' in row:
                            electrode_name = row['electrode_name']
                        elif 'electrode_number' in row:
                            electrode_name = f"E{int(row['electrode_number']):03d}"
                        else:
                            electrode_name = f"E{i+1:03d}"
                            
                        electrode_data = {
                            'electrode_number': i + 1,
                            'electrode_name': electrode_name,
                            'x': int(row['x']),
                            'y': int(row['y']),
                            'color': self.default_color
                        }
                        self.electrodes.append(electrode_data)
                        
                if self.electrodes:
                    # Update current electrode number for automatic mode
                    if self.insertion_mode == "automatic":
                        # Find the highest E-numbered electrode
                        e_electrodes = [e for e in self.electrodes if e['electrode_name'].startswith('E')]
                        if e_electrodes:
                            max_num = max(int(e['electrode_name'][1:]) for e in e_electrodes 
                                        if e['electrode_name'][1:].isdigit())
                            self.current_electrode_num = max_num + 1
                    
                self.update_status()
                print(f"Loaded {len(self.electrodes)} electrodes from {filename}")
            except Exception as e:
                print(f"Error loading data: {e}")
                
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key_Escape:
            self.selected_electrode_idx = None
            self.update_display()
            self.update_status()
        elif event.key() == Qt.Key_Delete:
            if self.selected_electrode_idx is not None:
                self.remove_electrode(self.selected_electrode_idx)
        elif event.key() == Qt.Key_Up:
            self.move_selected_electrode(0, -self.move_step)
        elif event.key() == Qt.Key_Down:
            self.move_selected_electrode(0, self.move_step)
        elif event.key() == Qt.Key_Left:
            self.move_selected_electrode(-self.move_step, 0)
        elif event.key() == Qt.Key_Right:
            self.move_selected_electrode(self.move_step, 0)
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_S:
                self.save_data()
                QMessageBox.information(self, "Saved", "Progress saved!")
            elif event.key() == Qt.Key_Z:
                if self.electrodes:
                    self.electrodes.pop()
                    self.selected_electrode_idx = None
                    self.update_display()
                    self.update_status()


class ImageWidget(QWidget):
    """Custom widget for image display and interaction"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
    def paintEvent(self, event):
        """Draw the image and electrodes"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Check if image is loaded
        if self.parent.original_image is None:
            painter.fillRect(self.rect(), QColor(50, 50, 50))
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.setFont(QFont('Arial', 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded\n\nPlease select an image file")
            return
        
        # Calculate scale to fit image in widget
        widget_size = self.size()
        image_size = self.parent.original_image.shape[:2]
        scale_x = widget_size.width() / image_size[1]
        scale_y = widget_size.height() / image_size[0]
        scale = min(scale_x, scale_y) * 0.95  # Leave some margin
        
        self.parent.scale_factor = scale
        
        # Calculate image position (centered)
        scaled_width = int(image_size[1] * scale)
        scaled_height = int(image_size[0] * scale)
        x_offset = (widget_size.width() - scaled_width) // 2
        y_offset = (widget_size.height() - scaled_height) // 2
        
        # Store offset for mouse coordinate conversion
        self.x_offset = x_offset
        self.y_offset = y_offset
        
        # Convert numpy array to QImage and draw
        height, width, channel = self.parent.original_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.parent.original_image.data, width, height, 
                        bytes_per_line, QImage.Format_RGB888)
        
        painter.drawImage(QRect(x_offset, y_offset, scaled_width, scaled_height), q_image)
        
        # Draw electrodes
        for i, electrode in enumerate(self.parent.electrodes):
            # Convert to display coordinates
            display_x = int(electrode['x'] * scale + x_offset)
            display_y = int(electrode['y'] * scale + y_offset)
            
            # Get color
            color = self.parent.color_map.get(electrode['color'], (255, 0, 0))
            is_selected = (i == self.parent.selected_electrode_idx)
            
            # Draw circle
            if is_selected:
                painter.setPen(QPen(QColor(255, 165, 0), 3))  # Orange for selected
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(display_x, display_y), 
                                  self.parent.circle_radius + 5, 
                                  self.parent.circle_radius + 5)
                
            painter.setPen(QPen(QColor(*color), 2))
            painter.setBrush(QBrush(QColor(*color)))
            painter.drawEllipse(QPoint(display_x, display_y), 
                              self.parent.circle_radius, 
                              self.parent.circle_radius)
            
            # Draw electrode name
            painter.setPen(QPen(QColor(0, 0, 0), 1))  # Black text
            painter.setFont(QFont('Arial', 8, QFont.Bold))  # Bold for better visibility
            text_rect = painter.fontMetrics().boundingRect(electrode['electrode_name'])
            text_x = display_x - text_rect.width() // 2
            text_y = display_y - self.parent.circle_radius - 5
            painter.drawText(text_x, text_y, electrode['electrode_name'])
            
    def mousePressEvent(self, event):
        """Handle mouse clicks"""
        # Check if image is loaded
        if self.parent.original_image is None:
            return
            
        # Convert to image coordinates
        x = int((event.x() - self.x_offset) / self.parent.scale_factor)
        y = int((event.y() - self.y_offset) / self.parent.scale_factor)
        
        # Check if click is within image bounds
        if (0 <= x < self.parent.original_image.shape[1] and 
            0 <= y < self.parent.original_image.shape[0]):
            
            if event.button() == Qt.LeftButton:
                # Check if clicking on existing electrode
                closest_idx = self.parent.find_closest_electrode(x, y)
                
                if closest_idx is not None:
                    # Select the electrode
                    self.parent.selected_electrode_idx = closest_idx
                    self.parent.update_display()
                    self.parent.update_status()
                else:
                    # Add new electrode
                    self.parent.add_electrode(x, y)
                    
            elif event.button() == Qt.RightButton:
                # Remove electrode
                closest_idx = self.parent.find_closest_electrode(x, y)
                if closest_idx is not None:
                    electrode_name = self.parent.electrodes[closest_idx]["electrode_name"]
                    reply = QMessageBox.question(self, 'Remove Electrode', 
                                               f'Remove electrode {electrode_name}?',
                                               QMessageBox.Yes | QMessageBox.No, 
                                               QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        self.parent.remove_electrode(closest_idx)


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Check if default image exists
    default_image = "GSN-256.png"
    initial_image = default_image if os.path.exists(default_image) else None
        
    # Create and show main window
    window = ElectrodeClickerQt(initial_image)
    window.show()
    
    # If no initial image, prompt user to select one
    if initial_image is None:
        window.current_image_label.setText("Please select an image to begin")
        QMessageBox.information(window, "Welcome", 
                              "Welcome to Electrode Clicker!\n\n"
                              "Please select an image file (.png or .jpg) to begin marking electrodes.")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 