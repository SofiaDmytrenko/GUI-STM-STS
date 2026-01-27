from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QMainWindow, QTableWidgetItem, QMenu,
    QTextEdit, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QShortcut, QKeySequence, QTextCursor, QTextCharFormat, QColor
from gui.helpers.styles import Style, Utils
import logging

logger = logging.getLogger(__name__)

class MetadataTab(QWidget):
    def __init__(self, channel_data, parent=None):
        super().__init__(parent)
        self.channel_data = channel_data

        main_layout = QVBoxLayout(self)

        # ----------------------------------------
        # Search layout
        # ----------------------------------------
        search_widget, self.search_box = Style.search_line()
        main_layout.addWidget(search_widget)

        self.search_box.returnPressed.connect(self.search_metadata)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search_box.setFocus)

        # Add to main layout
        main_layout.addWidget(search_widget)

        self.table = Style.metatab_table()
        self.table.cellClicked.connect(self.copy_metadata_cell)
        main_layout.addWidget(self.table)

    # ---------------------------------------------------------------------
    # Set metadata
    # ---------------------------------------------------------------------
    def set_metadata(self, attrs: dict):
        sorted_items = self.sort_metadata(attrs)
        self.fill_table(sorted_items)

    def fill_table(self, items):
        self.table.setRowCount(len(items))
        self.long_rows = set()

        for row, (key, value) in enumerate(items):
            key_item = Style.metatab_key_item(str(key))
            value_browser = Style.metatab_value_browser(str(value))

            self.table.setItem(row, 0, key_item)
            self.table.setCellWidget(row, 1, value_browser)

            value_browser.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            value_browser.customContextMenuRequested.connect(
                lambda pos, r=row: self.open_context_menu(r, pos)
            )

            empty_item = QTableWidgetItem(" ")
            empty_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, empty_item)

            QTimer.singleShot(0, lambda r=row, b=value_browser: self.adjust_row_height(r, b))

        self.table.horizontalHeader().sectionResized.connect(
            lambda idx, old, new: QTimer.singleShot(10, self.recalculate_metadata_row_heights)
        )

        QTimer.singleShot(30, self.recalculate_metadata_row_heights)

    # ---------------------------------------------------------------------
    # Search logic
    # ---------------------------------------------------------------------
    def search_metadata(self):
        query = self.search_box.text().strip()

        # Convert to lowercase for matching, but keep original for highlighting
        query_lc = query.lower()

        # If query empty - show all rows and remove highlights
        if not query:
            for row in range(self.table.rowCount()):
                self.table.setRowHidden(row, False)
                val_widget = self.table.cellWidget(row, 1)
                if isinstance(val_widget, QTextBrowser):
                    self.highlight_matches(val_widget, "")  # clear highlights
            return

        # Perform search on all rows
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_widget = self.table.cellWidget(row, 1)

            key_text = key_item.text() if key_item else ""
            val_text = val_widget.toPlainText() if val_widget else ""

            # Case-insensitive search
            match = (
                    query_lc in key_text.lower() or
                    query_lc in val_text.lower()
            )

            # Show or hide row
            self.table.setRowHidden(row, not match)

            # Highlight matches in value column
            if isinstance(val_widget, QTextBrowser):
                if match:
                    self.highlight_matches(val_widget, query)
                else:
                    self.highlight_matches(val_widget, "")

    def highlight_matches(self, browser: QTextBrowser, pattern: str):
        doc = browser.document()

        # Clear previous formatting
        cursor = QTextCursor(doc)
        default_fmt = QTextCharFormat()

        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.setCharFormat(default_fmt)
        cursor.clearSelection()
        cursor.endEditBlock()

        if not pattern:
            return

        # Apply highlighting to each match
        highlight_fmt = QTextCharFormat()
        highlight_fmt.setBackground(QColor("yellow"))

        cursor = doc.find(pattern)
        while not cursor.isNull():
            cursor.mergeCharFormat(highlight_fmt)
            cursor = doc.find(pattern, cursor)

    def add_expand_button(self, row):
        btn = Style.create_expand_button()
        btn.clicked.connect(lambda: self.open_full_view(row))
        self.table.setCellWidget(row, 2, btn)

    def open_context_menu(self, row, pos: QPoint):
        menu = QMenu()

        # Default actions (copy, select all)
        browser = self.table.cellWidget(row, 1)
        default_menu = browser.createStandardContextMenu()
        for act in default_menu.actions():
            menu.addAction(act)

        menu.addSeparator()
        action_search = menu.addAction("Search the row")
        action = menu.exec(browser.mapToGlobal(pos))

        if action == action_search:
            self.open_full_view(row)

    def open_full_view(self, row):
        key_item = self.table.item(row, 0)
        key_text = key_item.text() if key_item else "(no key)"

        browser = self.table.cellWidget(row, 1)
        value = browser.toPlainText()

        dlg = QMainWindow(self)
        dlg.setWindowTitle(f"Search: {key_text}")
        central = QWidget()
        dlg.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Search bar
        search_widget, search_box, search_count_field, search_count_label, prev_btn, next_btn = Style.full_view_search()
        layout.addWidget(search_widget)

        QShortcut(QKeySequence("Ctrl+F"), dlg, activated=lambda: search_box.setFocus())

        # Text area
        text_viewer = QTextEdit()
        text_viewer.setReadOnly(True)
        text_viewer.setPlainText(value)
        text_viewer.setStyleSheet(Style.SCROLLBAR_VERTICAL)
        layout.addWidget(text_viewer)

        # --- Search & highlight & navigation logic ---
        # Stores search data
        search_state = {
            "pattern": "",
            "positions": [],  # list of start offsets (match)
            "index": 0  # current match index
        }

        def compute_all_matches(pattern: str):
            if not pattern:
                return []

            doc = text_viewer.document()
            cur = doc.find(pattern)
            positions = []

            while not cur.isNull():
                positions.append(cur.selectionStart())
                cur = doc.find(pattern, cur)

            return positions

        def highlight_all(pattern: str):
            doc = text_viewer.document()

            # Clear previous highlights
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.setCharFormat(QTextCharFormat())

            if not pattern:
                return

            fmt = QTextCharFormat()
            fmt.setBackground(QColor("yellow"))

            cur = doc.find(pattern)
            while not cur.isNull():
                cur.mergeCharFormat(fmt)
                cur = doc.find(pattern, cur)

        # --- jump tp 3rd line from top ---
        def jump_to_index(idx: int):
            if not search_state["positions"]:
                return

            doc = text_viewer.document()
            pattern = search_state["pattern"]
            pos = search_state["positions"][idx]

            # Create a cursor that selects the found match
            cur = QTextCursor(doc)
            cur.setPosition(pos)
            cur.movePosition(QTextCursor.MoveOperation.Right,
                             QTextCursor.MoveMode.KeepAnchor,
                             len(pattern))
            text_viewer.setTextCursor(cur)

            # Compute line number for the match start
            cur2 = QTextCursor(doc)
            cur2.setPosition(pos)
            block = cur2.block()
            block_number = block.blockNumber()

            # Approximate pixel height of a line and calculate target scrollbar value
            fm = text_viewer.fontMetrics()
            line_h = fm.lineSpacing()
            # desired top block number = block_number - 2 (3rd line)
            top_block = max(0, block_number - 2)
            desired_scroll_px = top_block * line_h

            vs = text_viewer.verticalScrollBar()
            # clamp to valid range
            desired_scroll_px = max(0, min(desired_scroll_px, vs.maximum()))
            vs.setValue(int(desired_scroll_px))

            # keep cursor visible / selected
            text_viewer.ensureCursorVisible()
            # update number field
            search_count_field.setText(str(search_state["index"] + 1))

        def find_next():
            if not search_state["positions"]:
                return
            search_state["index"] = (search_state["index"] + 1) % len(search_state["positions"])
            jump_to_index(search_state["index"])

        def find_prev():
            if not search_state["positions"]:
                return
            search_state["index"] = (search_state["index"] - 1) % len(search_state["positions"])
            jump_to_index(search_state["index"])

        # --- manual index entry ---
        def go_to_manual_index():
            if not search_state["positions"]:
                return
            try:
                val = int(search_count_field.text())
            except ValueError:
                return
            if not (1 <= val <= len(search_state["positions"])):
                return
            search_state["index"] = val - 1
            jump_to_index(search_state["index"])

        def run_search():
            pattern = search_box.text().strip()
            search_state["pattern"] = pattern
            search_state["positions"] = compute_all_matches(pattern)
            search_state["index"] = 0
            highlight_all(pattern)

            count = len(search_state["positions"])
            search_count_label.setText(f"{count if count else ''}")
            if count:
                search_count_field.setText("1")
                jump_to_index(0)
            else:
                search_count_field.setText("0")

        # --- Connections and shortcuts ---
        search_box.returnPressed.connect(run_search)
        search_count_field.returnPressed.connect(go_to_manual_index)
        next_btn.clicked.connect(find_next)
        prev_btn.clicked.connect(find_prev)

        for key in [Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Up, Qt.Key.Key_Down]:
            shortcut = QShortcut(QKeySequence(key), dlg)
            if key in [Qt.Key.Key_PageUp, Qt.Key.Key_Up]:
                shortcut.activated.connect(find_prev)
            else:
                shortcut.activated.connect(find_next)

        dlg.resize(700, 400)
        dlg.show()

    # ---------------------------------------------------------------------
    # Row height logic
    # ---------------------------------------------------------------------
    def recalculate_metadata_row_heights(self):
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 1)
            if widget is not None:
                self.adjust_row_height(row, widget)

    def adjust_row_height(self, row, browser):
        height = Utils.compute_textedit_height(
            browser,
            self.table.columnWidth(1),
            max_height=300,
            overhead=20
        )

        text = browser.toPlainText()
        lines = text.count("\n") + 1

        if lines <= 2:
            fm = browser.fontMetrics()
            height = max(24, fm.lineSpacing() * lines + 6)

        browser.setFixedHeight(height)
        self.table.setRowHeight(row, height)

        is_long = (lines > 2) or (height >= 300) or (len(text) > 100)
        if is_long:
            self.long_rows.add(row)
            self.add_expand_button(row)
        else:
            self.long_rows.discard(row)
            self.table.removeCellWidget(row, 2)

    # ---------------------------------------------------------------------
    # Sorting
    # ---------------------------------------------------------------------
    def sort_metadata(self, attrs: dict) -> list:
        keys = list(attrs.keys())
        sorted_keys = []

        # RHK_SessionText first
        session_key = "RHK_SessionText"
        if session_key in keys:
            sorted_keys.append(session_key)
            keys.remove(session_key)

        # Move long values (>4 lines) to the end
        long_value_keys = [
            k for k in keys
            if isinstance(attrs[k], str) and len(attrs[k].splitlines()) > 4
        ]
        keys = [k for k in keys if k not in long_value_keys]

        sorted_keys.extend(keys)
        sorted_keys.extend(long_value_keys)

        return [(k, attrs[k]) for k in sorted_keys]

    # ---------------------------------------------------------------------
    # Copying cell logic
    # ---------------------------------------------------------------------
    def copy_metadata_cell(self, row, column):
        if column != 0:
            return

        browser = self.table.cellWidget(row, 1)
        if not browser:
            return

        Utils.copy_text_with_feedback(self, browser.toPlainText())
