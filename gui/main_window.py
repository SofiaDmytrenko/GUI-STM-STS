from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QToolButton, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
from gui.data_parser import DataManager
from gui.channel_viewers.sts_viewer_folder.sts_viewer import STSViewer
from gui.channel_viewers.topo_viewer import TopoViewer
from gui.helpers.styles import Style, Utils, ConditionalDelegate
import logging

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer_windows = []
        self.data_manager = DataManager()

        # --- Window setup ---
        self.setWindowTitle("STM/STS Data Viewer")
        self.resize(750, 500)
        self.setMinimumSize(500, 250)

        # --- Central widget ---
        central_widget = QWidget()
        central_widget.setStyleSheet("QWidget { background-color: white; }")
        self.setCentralWidget(central_widget)

        # --- Splitter for left/right sections ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- Left: Metadata title, Metadata table ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(0)

        # Metadata title
        lbl_metadata_title = QLabel("| METADATA")
        Style.main_window_title(lbl_metadata_title)
        left_layout.addWidget(lbl_metadata_title)

        # Metadata table
        frame, self.table_metadata = Style.metadata_table()
        self.table_metadata.cellClicked.connect(self.copy_metadata_cell)
        self.table_metadata.horizontalHeader().sectionResized.connect(
            lambda *_: QTimer.singleShot(10, self.recalculate_metadata_row_heights)
        )

        left_layout.addWidget(frame)
        left_widget.setMinimumWidth(300)

        # --- Right: Channel title, Channel list, Load button ---
        right_widget = QWidget()
        right_widget.setMinimumWidth(150)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(0)

        # Channel title
        lbl_channel_title = QLabel("| CHANNELS")
        Style.main_window_title(lbl_channel_title)
        right_layout.addWidget(lbl_channel_title)

        # Channel list
        self.channel_list = HoverListWidget()
        Style.channel_list(self.channel_list)

        self.channel_list.setMouseTracking(True)
        self.channel_list.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self.channel_list.entered.connect(
            lambda index: self.channel_list.viewport().setCursor(Qt.CursorShape.PointingHandCursor))
        self.channel_list.viewport().leaveEvent = lambda event: self.channel_list.viewport().setCursor(
            Qt.CursorShape.ArrowCursor)
        self.channel_list.itemDoubleClicked.connect(self.open_correct_viewer)

        right_layout.addWidget(self.channel_list, 1)
        right_layout.addSpacing(6)

        # Load button
        self.btn_load = QToolButton()
        self.btn_load.setText("Open File")
        self.btn_load.clicked.connect(self.open_file)
        Style.load_button(self.btn_load)
        right_layout.addWidget(self.btn_load)

        # --- Add both sides to splitter ---
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([550, 200])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        Style.style_splitter(splitter)

        # --- Layout for central widget ---
        layout = QVBoxLayout(central_widget)
        layout.addWidget(splitter)

    # ------------------------- Open File -------------------------
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Data Files (*.sm4 *.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            self.data_manager.load_file(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return

        # Update metadata table
        metadata_items = self.data_manager.get_metadata_items()
        self.update_metadata(metadata_items)

        # Populate channel list
        self.channel_list.clear()
        channels = self.data_manager.get_channels()

        if not channels:
            QMessageBox.warning(self, "No data", "No channels found in this file.")
            return

        for name, title in channels:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.channel_list.addItem(item)

    # ------------------------- Viewer handling -------------------------
    def open_correct_viewer(self):
        selected_item = self.channel_list.currentItem()
        if not selected_item:
            return

        channel_name = selected_item.data(Qt.ItemDataRole.UserRole)
        if not channel_name:
            return

        channel_data = self.data_manager.get_channel_data(channel_name)
        if not channel_data:
            QMessageBox.warning(self, "Warning", "No data found for this channel.")
            return

        ch_type = channel_data.get("type", "").lower()
        try:
            if ch_type == "sts":
                self.open_sts(channel_name, channel_data)
            elif ch_type == "topo":
                self.open_topo(channel_name, channel_data)
            else:
                QMessageBox.warning(self, "Unknown channel type", f"Unhandled channel type: {ch_type}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open viewer:\n{e}")

    def open_sts(self, channel_name: str, channel_data: dict):
        all_channels = getattr(self.data_manager.current_data, "channels", {}) or {}

        # Reference channel
        current_ch = None
        for ch in all_channels.values():
            t = ch.get("title", "").lower().strip()
            if t == "current" and current_ch is None:
                current_ch = ch

        filename = channel_data.get("attrs", {}).get("CSV filename") or channel_data.get("attrs", {}).get("File")

        viewer = STSViewer(
            channel_data=channel_data,
            current_channel=current_ch,
            parent=None
        )
        viewer.setWindowTitle(f"STS Viewer ({channel_name}) - {filename}")
        viewer.show()
        self.viewer_windows.append(viewer)

    def open_topo(self, channel_name: str, channel_data: dict):
        filename = channel_data.get("attrs", {}).get("CSV filename") or channel_data.get("attrs", {}).get("File")
        viewer = TopoViewer(channel_data, parent=None)
        viewer.setWindowTitle(f"Topo Viewer ({channel_name}) – {filename}")
        viewer.show()
        self.viewer_windows.append(viewer)

    # ------------------------- Metadata table -------------------------
    def update_metadata(self, items):
        self.table_metadata.setRowCount(len(items))
        for row, (key, value) in enumerate(items):
            key_item = Style.metadata_key_column(key)
            browser = Style.metadata_value_column(value)
            self.table_metadata.setItem(row, 0, key_item)
            self.table_metadata.setCellWidget(row, 1, browser)

        # Recalculate row heights
        QTimer.singleShot(30, self.recalculate_metadata_row_heights)
        self.table_metadata.setItemDelegate(ConditionalDelegate())

    def recalculate_metadata_row_heights(self):
        for row in range(self.table_metadata.rowCount()):
            widget = self.table_metadata.cellWidget(row, 1)
            if widget is not None:
                self.adjust_row_height(row, widget)

    def adjust_row_height(self, row, browser):
        height = Utils.compute_textedit_height(
            browser,
            self.table_metadata.columnWidth(1),
            max_height=800,
            overhead=16,
            document_margin=0
        )

        browser.setMinimumHeight(height)
        browser.setMaximumHeight(800)
        self.table_metadata.setRowHeight(row, height)

    def copy_metadata_cell(self, row, column):
        text = ""

        if column == 0:
            item = self.table_metadata.item(row, column)
            if item:
                text = item.text()
        elif column == 1:
            browser = self.table_metadata.cellWidget(row, column)
            if browser:
                text = browser.toPlainText()

        Utils.copy_text_with_feedback(self, text)


class HoverListWidget(QListWidget):
    def mouseMoveEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)
