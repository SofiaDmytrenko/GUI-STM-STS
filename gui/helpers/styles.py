from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy, QTableWidget,
    QHeaderView, QStyledItemDelegate, QAbstractItemView, QFrame,
    QTableWidgetItem, QTextBrowser, QTextEdit, QLineEdit, QLabel,
    QToolButton, QApplication, QGraphicsOpacityEffect, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QPropertyAnimation
from PyQt6.QtGui import QFont, QPen, QColor, QCursor

class Style:
    # ---------------------- Stylesheets ----------------------
    MAIN_WINDOW_METADATA = """
        QTableWidget {
            background: transparent;
            border-radius: 0;
            border: none;
        }
        QTableWidget::item {
            background: transparent;
            border: none;
            padding-left: 6px;
            padding-right: 6px;
        }
        QHeaderView::section {
            background-color: #e8e8e8;
            border: none;
            padding: 4px;
        }"""

    METADATA_FRAME = """
        QFrame {
            border: 2px solid #cccccc;
            border-radius: 4px;
            background: transparent;
        }"""

    METADATA_VALUE_COLUMN = """
        QTextBrowser {
            background: transparent;
            border: none;
            outline: none;
            padding-top: 8px;
            padding-bottom: 8px;
        }"""

    CHANNELS_LIST = """
        QListWidget {
            background-color: #dbdbdb;
        }
        QListWidget::viewport {
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 6px 12px;
            background-color: white;
            color: black;
            border-radius: 4px;
            border: 1px solid #333333;
        }
        QListWidget::item:hover {
            background-color: #f3f3f3;
        }"""

    COPY_LABEL = """
        QLabel {
            background-color: white;
            color: black;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #cccccc;
            font-size: 12px;
        }"""

    LOAD_BUTTON = """
        QToolButton {
            font-size: 14px;
            font-weight: bold;
            color: white;
            background-color: #242424;
            border-radius: 6px;
            padding: 6px 12px;
            }
        QToolButton:hover {
            background-color: #212121;
        }"""

    TOOLBAR_FRAME = """
        QFrame {
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }"""

    SCROLLBAR_VERTICAL = """
        QScrollBar:vertical {
            width: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line,
        QScrollBar::sub-line {
            height: 0;
        }"""

    BUTTON = """
        QToolButton {
            border: 1px solid #a0a0a0;
            border-radius: 6px;
            background-color: #fafafa;
            padding: 2px 6px;
        }
        QToolButton:hover {
            background-color: #f3f3f3;
        }
        QToolButton:pressed {
            background-color: #e8e8e8;
        }"""

    CHECKLIST_ICON = """
        QToolButton {
            padding: 0;
            border-radius: 6px;
            background-color: #f8f8f8;
            border: 1px solid #cccccc;
            font-size: 18px;
        }
        QToolButton:hover {
            background-color: #e6e6e6;
        }
        QToolButton:pressed {
            background-color: #dcdcdc;
        }"""

    ICON_BUTTON = """
        QToolButton {
            border: none;
            background: transparent;
            font-size: 18px;
        }"""

    SECTION_TITLE = """
        QLabel {
            border: 2px solid #dbdbdb;
            border-left: none;
            border-right: none;
        }"""

    SPLITTER = """
        QSplitter::handle {
            background-color: #e8e8e8;
            border-radius: 2px;
        }"""

    PEAK_PANEL = """
        QWidget#peakPanel {
            border-radius: 4px;
            border: 1px solid #cccccc;
        }"""

    PANEL = """
        QWidget#rightPanel {
            background-color: #fafafa;
            border-radius: 4px;
            border: 1px solid #cccccc;
            padding: 6px;
        }"""

    CURVE_LIST = """
        QListWidget::item {
            padding: 3px 5px 3px 5px;
            border-radius: 4px;
            background-color: #f0f0f0;
        }
        QListWidget::item:hover {
            background-color: #e8e8e8;
        }"""

    TOPO_LABEL = """
        QLabel {
            font-size: 13px;
            padding: 8px 4px 8px 4px;
            background-color: white;
            border-radius: 6px;
            border: 1px solid #cccccc;
        }"""

    TOPO_LABEL_CONTAINER = """
        QWidget {
            background-color: white;
            border-radius: 6px;
            border: 1px solid #cccccc;
        }
        QLabel {
            border: none;
            border-radius: 0;
        }"""

    CMAP_COMBOBOX = """
        QComboBox {
            border: 1px solid #cccccc;
            border-radius: 6px;
            padding-left: 10px;
            padding-right: 24px;
            font-size: 14px;
        }
        QComboBox:hover {
            background-color: #f3f3f3;
        }
        QComboBox QAbstractItemView {
            background-color: #f0f0f0;
            outline: none;
            selection-color: black;
            padding: 2px 0 2px 2px;
            border-radius: 6px;
        }
        QComboBox QAbstractItemView::item {
            background-color: white;
            border-bottom: 1px solid #dbdbdb;
            border-radius: 2px;
            margin: 1px 4px;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #f3f3f3;
        }"""

    SMOOTH_SLIDER = """
        QSlider::groove:horizontal {
            height: 16px;
            border-radius: 5px;
        }
        QSlider::handle:horizontal {
            background: #fafafa;
            border: 2px solid #bfbfbf;
            width: 16px;
            margin: -2px 0;
            border-radius: 5px;
        }
        QSlider::handle:horizontal:hover {
            background: #f2f2f2;
        }
        QSlider::sub-page:horizontal {
            background: #808080;
            border-radius: 5px;
        }
        QSlider::add-page:horizontal {
            background: #d9d9d9;
            border-radius: 5px;
        }"""

    SMOOTH_LABEL = """
        QLabel {
            border: none;
            font-size: 14px;
            padding-right: 2px;
        }"""

    MENU = """
        QMenu {
            background-color: #f8f8f8;
            border: 1px solid #cccccc;
            padding: 4px;
        }
        QMenu::item {
            padding: 2px 22px 2px 16px;
            font-size: 12px;
        }
        QMenu::item:selected {
            background-color: #f3f3f3;
        }
        QMenu::item:pressed {
            background-color: #e8e8e8;
        }"""

    # ---------------------- Style methods ----------------------
    # --- Main Window ---
    @staticmethod
    def main_window_title(title):
        font = QFont()
        font.setPointSize(12)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        title.setFont(font)
        title.setContentsMargins(2, 0, 0, 6)

    @staticmethod
    def metadata_table():
        frame = QFrame()
        frame.setStyleSheet(Style.METADATA_FRAME)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        table.setShowGrid(False)
        table.setHorizontalHeader(CustomHeader(Qt.Orientation.Horizontal, table))
        table.verticalHeader().setVisible(False)

        table.setFrameShape(QFrame.Shape.NoFrame)
        table.viewport().setAutoFillBackground(False)
        table.viewport().setStyleSheet("background: transparent;")

        table.setColumnWidth(0, 130)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setStretchLastSection(True)

        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setStyleSheet(Style.MAIN_WINDOW_METADATA + Style.SCROLLBAR_VERTICAL)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.addWidget(table)

        return frame, table

    @staticmethod
    def metadata_key_column(key: str) -> QTableWidgetItem:
        item = QTableWidgetItem(str(key))
        f = item.font()
        f.setBold(True)
        item.setFont(f)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return item

    @staticmethod
    def metadata_value_column(value: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setAcceptRichText(False)
        browser.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        browser.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        browser.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        browser.setFrameShape(QFrame.Shape.NoFrame)
        browser.setFrameShadow(QFrame.Shadow.Plain)
        browser.setPlainText(str(value))
        browser.document().setDocumentMargin(0)
        browser.setStyleSheet(Style.METADATA_VALUE_COLUMN)
        return browser

    @staticmethod
    def channel_list(channel_list):
        channel_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        channel_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        channel_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        channel_list.setSpacing(4)
        channel_list.setViewportMargins(0, 4, 0, 0)
        channel_list.setStyleSheet(Style.CHANNELS_LIST + Style.SCROLLBAR_VERTICAL)

    @staticmethod
    def load_button(load_button):
        load_button.setFixedHeight(50)
        load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        load_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        load_button.setStyleSheet(Style.LOAD_BUTTON)

    # --- Toolbar ---
    @staticmethod
    def create_toolbar_frame():
        frame = QFrame()
        frame.setFixedHeight(44)
        frame.setStyleSheet(Style.TOOLBAR_FRAME)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 1, 5, 1)
        layout.setSpacing(4)
        return layout, frame

    @staticmethod
    def style_buttons(buttons):
        for btn in buttons:
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(Style.BUTTON)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def style_menu(menu):
        menu.setStyleSheet(Style.MENU)
        menu.setCursor(Qt.CursorShape.PointingHandCursor)

    @staticmethod
    def style_cmap_combo(cmap_label, cmap_combo):
        cmap_label.setStyleSheet("QLabel { border: none; font-size: 13px; }")
        cmap_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        cmap_combo.setFixedHeight(28)
        cmap_combo.setStyleSheet(Style.CMAP_COMBOBOX + Style.SCROLLBAR_VERTICAL)

    @staticmethod
    def style_smooth_widgets(label, slider, value_label):
        label.setStyleSheet(Style.SMOOTH_LABEL)
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        slider.setMinimumWidth(50)
        slider.setFixedHeight(26)
        slider.setStyleSheet(Style.SMOOTH_SLIDER + Style.SCROLLBAR_VERTICAL)

        value_label.setMinimumWidth(42)
        value_label.setStyleSheet(Style.SMOOTH_LABEL)

    @staticmethod
    def style_checkbox(checkbox):
        checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        checkbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        checkbox.setStyleSheet("QCheckBox { padding-left: 2px; font-size: 14px; } ")

    @staticmethod
    def create_pixel_divider():
        divider = QFrame()
        divider.setStyleSheet("QFrame { background-color: #cccccc; border: none; }")
        divider.setFixedSize(1, 28)
        divider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(divider)
        layout.addStretch()
        return wrapper

    # --- Panels ---
    @staticmethod
    def style_canvas(canvas):
        canvas.setMinimumWidth(400)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return canvas

    @staticmethod
    def style_peak_panel(peak_panel):
        peak_panel.setObjectName("peakPanel")
        peak_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        peak_panel.setStyleSheet(Style.PEAK_PANEL)

    @staticmethod
    def style_right_panel(right_panel):
        right_panel.setObjectName("rightPanel")
        right_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_panel.setStyleSheet(Style.PANEL)

    @staticmethod
    def style_splitter(splitter):
        splitter.setHandleWidth(8)
        splitter.setStyleSheet(Style.SPLITTER)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # --- Metadata Tab ---
    @staticmethod
    def search_line():
        search_box = QLineEdit()
        search_icon = QLabel("🔎")
        search_icon.setStyleSheet("QLabel { font-size: 16px; }")
        search_icon.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        search_box.setPlaceholderText("Search metadata...")
        search_box.setToolTip("Press [Enter] to search")
        search_box.setClearButtonEnabled(True)
        search_box.setMinimumHeight(32)

        layout.addWidget(search_icon)
        layout.addWidget(search_box)

        return widget, search_box

    @staticmethod
    def metatab_table():
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Key", "Value", ""])
        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setStretchLastSection(False)

        table.setColumnWidth(0, 200)
        table.setColumnWidth(2, 32)

        table.setShowGrid(False)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.verticalScrollBar().setSingleStep(5)

        table.setStyleSheet(
            "QTableWidget::item { background: transparent; }" + Style.SCROLLBAR_VERTICAL
        )

        table.setItemDelegate(BorderDelegate())

        return table

    @staticmethod
    def metatab_key_item(key_str: str) -> QTableWidgetItem:
        item = QTableWidgetItem(key_str)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return item

    @staticmethod
    def metatab_value_browser(value_str: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setAcceptRichText(False)
        browser.setOpenLinks(False)
        browser.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setStyleSheet("""
            QTextBrowser {
                padding: 2px;
                border: none;
                selection-background-color: #a8d1ff;
                selection-color: black;
                text-decoration: none;
            }
        """)
        browser.setPlainText(value_str)
        return browser

    @staticmethod
    def create_expand_button():
        btn = QToolButton()
        btn.setText("📖")
        Style.icon_button(btn)
        return btn

    @staticmethod
    def full_view_search():
        search_widget, search_box = Style.search_line()

        layout = search_widget.layout()
        # layout.setSpacing(6)

        # Count field
        count_field = QLineEdit()
        count_field.setPlaceholderText("0")
        count_field.setFixedSize(32, 32)
        count_field.setStyleSheet("""
                QLineEdit {
                    background: transparent;
                    border: none;
                    border-bottom: 1px solid #dbdbdb;
                }
            """)
        layout.addWidget(count_field)

        # Slash
        slash_label = QLabel("/")
        slash_label.setFixedHeight(32)
        slash_label.setStyleSheet("QLabel { font-size: 14px; }")
        layout.addWidget(slash_label)

        # Count label
        count_label = QLabel("")
        count_label.setFixedSize(32, 32)
        Style.font_bold(count_label)
        layout.addWidget(count_label)

        # Navigation buttons
        prev_btn = QToolButton()
        prev_btn.setText("🔼")
        Style.style_checklist_icon(prev_btn)

        next_btn = QToolButton()
        next_btn.setText("🔽")
        Style.style_checklist_icon(next_btn)

        layout.addWidget(prev_btn)
        layout.addWidget(next_btn)

        return search_widget, search_box, count_field, count_label, prev_btn, next_btn

    @staticmethod
    def icon_button(btn):
        btn.setFixedSize(30, 30)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(Style.ICON_BUTTON)

    # Checklist
    @staticmethod
    def style_checklist_icon(chk_icon):
        chk_icon.setFixedSize(31, 31)
        chk_icon.setStyleSheet(Style.CHECKLIST_ICON)
        chk_icon.setCursor(Qt.CursorShape.PointingHandCursor)

    @staticmethod
    def style_title(label):
        font = QFont()
        font.setPointSize(10)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        label.setFont(font)
        label.setContentsMargins(2, 6, 0, 6)
        label.setWordWrap(True)
        label.setStyleSheet(Style.SECTION_TITLE)

    @staticmethod
    def style_curve_list(curve_list):
        curve_list.setMinimumWidth(100)
        curve_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        curve_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        curve_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        curve_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        curve_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        curve_list.verticalScrollBar().setSingleStep(5)
        curve_list.setSpacing(1)
        curve_list.setStyleSheet(Style.CURVE_LIST + Style.SCROLLBAR_VERTICAL)

    @staticmethod
    def style_input_label(input_label):
        input_font = QFont()
        input_font.setPointSize(8)
        input_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        input_label.setFont(input_font)
        input_label.setContentsMargins(4, 0, 0, 0)

    @staticmethod
    def font_bold(text):
        bold = QFont()
        bold.setBold(True)
        text.setFont(bold)

    # --- Fit Gauss ---
    @staticmethod
    def peak_coords_detected(peaks: list[float]) -> QPlainTextEdit:
        coords = QPlainTextEdit()
        coords.setReadOnly(True)
        coords.setPlainText("\n".join(f"{i + 1})   {p:.4f}" for i, p in enumerate(peaks)))
        coords.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        coords.setFrameStyle(QFrame.Shape.NoFrame)

        coords.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        coords.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        coords.setViewportMargins(4, 4, 0, 0)

        coords.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
        """)

        return coords

    @staticmethod
    def peak_row_manual(index: int, value: float, is_first: bool) -> tuple[QHBoxLayout, QLineEdit]:
        ROW_HEIGHT = 24
        LABEL_WIDTH = 20

        row = QHBoxLayout()
        row.setSpacing(2)
        row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        row.setContentsMargins(6, 8 if is_first else 1, 6, 0)

        lbl = QLabel(f"{index})")
        lbl.setFixedSize(LABEL_WIDTH, ROW_HEIGHT)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setContentsMargins(0, -2, 0, 0)

        le = QLineEdit(f"{value:.4f}")
        le.setFixedHeight(ROW_HEIGHT)
        le.setStyleSheet("""
            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 2px 4px;
                background-color: #fff;
            }
            QLineEdit:focus {
                border: 1px solid #999999;
            }
        """)

        row.addWidget(lbl)
        row.addWidget(le)

        return row, le

# ------------------------- Helpers -------------------------
class Utils:
    # --- Metadata utils ---
    @staticmethod
    def compute_textedit_height(
            browser: QTextEdit,
            column_width: int,
            *,
            min_height=24,
            max_height=300,
            overhead=16,
            document_margin=4
    ) -> int:
        doc = browser.document()
        doc.setDocumentMargin(document_margin)
        doc.setTextWidth(column_width - 10)

        doc_height = doc.size().height()
        height = int(doc_height) + overhead

        return max(min_height, min(max_height, height))

    @staticmethod
    def copy_text_with_feedback(parent: QWidget, text: str,
                                timeout=600, fade_duration=200):
        if not text:
            return

        QApplication.clipboard().setText(text)

        max_len = 30
        lines = text.split("\n")
        preview = "\n".join(lines[:2])
        longer = len(lines) > 2 or len(preview) > max_len

        if len(preview) > max_len:
            preview = preview[:max_len]

        display_text = preview + ("…" if longer else "")
        Utils.show_temporary_message(
            parent,
            f'"{display_text}" copied to clipboard',
            timeout,
            fade_duration
        )

    @staticmethod
    def show_temporary_message(parent: QWidget, text,
                               timeout=600, fade_duration=200):
        label = QLabel(text, parent)
        label.setStyleSheet(Style.COPY_LABEL)
        label.adjustSize()

        cursor_pos = QCursor.pos()
        widget_pos = parent.mapFromGlobal(cursor_pos)
        label.move(widget_pos + QPoint(10, 10))
        label.show()

        effect = QGraphicsOpacityEffect(label)
        label.setGraphicsEffect(effect)
        effect.setOpacity(1.0)

        def start_fade():
            anim = QPropertyAnimation(effect, b"opacity", label)
            anim.setDuration(fade_duration)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.finished.connect(label.deleteLater)
            anim.start()
            label._animation = anim

        QTimer.singleShot(timeout, start_fade)

class CustomHeader(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(False)

    def paintSection(self, painter, rect, logicalIndex):
        super().paintSection(painter, rect, logicalIndex)

        # Draw inner right border for header
        if logicalIndex < self.model().columnCount() - 1:
            pen = QPen(QColor("#cccccc"))
            painter.setPen(pen)
            x = rect.right() - 2
            painter.drawLine(x, rect.top(), x, rect.bottom())

class ConditionalDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        # Prepare pen
        pen = QPen(QColor("#e8e8e8"))
        pen.setWidth(1)
        painter.setPen(pen)

        row = index.row()
        col = index.column()
        row_count = index.model().rowCount()
        col_count = index.model().columnCount()
        rect = option.rect

        # Draw right border if not last column
        if col < col_count - 1:
            painter.drawLine(rect.topRight(), rect.bottomRight())

        # Draw bottom border if not last row
        if row < row_count - 1:
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

class BorderDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        # Draw horizontal line under each row
        painter.setPen(QColor("#ddd"))
        rect = option.rect
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        # Draw vertical line between col 0 and 1
        if index.column() == 0:
            painter.drawLine(rect.topRight(), rect.bottomRight())
