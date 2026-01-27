from PyQt6.QtWidgets import (
    QToolButton, QCheckBox, QMenu, QSizePolicy, QWidget, QVBoxLayout
)
from gui.helpers.styles import Style

class STSToolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---------------------- Buttons ----------------------
        self.create_buttons()
        self.create_menus()
        self.style_buttons()
        self.style_menu()
        self.create_layout()

    # ---------------------- Create buttons ----------------------
    def create_buttons(self):
        # Save
        self.btn_save = QToolButton()
        self.btn_save.setText("💾 Save ")
        self.btn_save.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_save.clicked.connect(self.btn_save.showMenu)

        # Normalize over I/U (button)
        # self.btn_normalize_over_IU = QToolButton()
        # self.btn_normalize_over_IU.setText("🔨 Normalize by I/U")

        # Normalize (menu)
        self.btn_normalize = QToolButton()
        self.btn_normalize.setText("🛠️ Normalize")
        self.btn_normalize.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_normalize.clicked.connect(self.btn_normalize.showMenu)

        # Filter
        self.btn_filter = QToolButton()
        self.btn_filter.setText("🫧 Filter ")
        self.btn_filter.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_filter.clicked.connect(self.btn_filter.showMenu)

        # Math / derivative
        self.btn_derivative = QToolButton()
        self.btn_derivative.setText("🧮 d/dU")

        self.btn_subtract = QToolButton()
        self.btn_subtract.setText("➖ Subtract")

        self.btn_divide = QToolButton()
        self.btn_divide.setText("➗ Divide")

        self.btn_average = QToolButton()
        self.btn_average.setText("📊 Avg")

        self.chk_swap_order = QCheckBox("Switch")

        # Fit
        self.btn_fit = QToolButton()
        self.btn_fit.setText("✨ Fit Gauss")

        # Update / hide / reset
        self.btn_update = QToolButton()
        self.btn_update.setText("📈")

        self.btn_chk_all = QToolButton()
        self.btn_chk_all.setText("✅")

        self.btn_hide = QToolButton()
        self.btn_hide.setText("🧹")

        self.btn_reset = QToolButton()
        self.btn_reset.setText("🗑️")

        # ---------------------- Tooltips ----------------------
        self.btn_save.setToolTip(
            "<html>"
            "Save chart image (.png) or<br>"
            "export data (.csv)"
            "</html>"
        )

        self.btn_normalize.setToolTip(
            "<html>"
            "Normalize curve by dividing it<br>"
            "by I/U or by I<sub>max</sub><br>"
            "<i>– Select one or more curves –</i>"
            "</html>"
        )

        self.btn_derivative.setToolTip(
            "<html>"
            "Compute derivative dI/dU or d<sup>2</sup>I/dU<sup>2</sup><br>"
            "<i>– Select one or more curves –</i>"
            "</html>"
        )

        self.btn_filter.setToolTip(
            "<html>"
            "Apply filter to reduce noise and smooth curves<br>"
            "<i>– Select one or more curves –</i>"
            "</html>"
        )

        self.btn_fit.setToolTip(
            "<html>"
            "Fit selected curve using Gaussian function<br>"
            "<i>– Select one curve –</i>"
            "</html>"
        )

        self.btn_subtract.setToolTip(
            "<html>"
            "Subtract one selected curve from another<br>"
            "<i>– Select two curves –</i>"
            "</html>"
        )

        self.btn_divide.setToolTip(
            "<html>"
            "Divide one selected curve by another<br>"
            "<i>– Select two curves –</i>"
            "</html>"
        )

        self.chk_swap_order.setToolTip(
            "<html>"
            "Swap the order of curves for "
            "<i>Subtract</i>/<i>Divide</i> operations"
            "</html>"
        )

        self.btn_average.setToolTip(
            "<html>"
            "Compute average of selected curves<br>"
            "<i>– Select two or more curves –</i>"
            "</html>"
        )

        self.btn_update.setToolTip("Update chart, plot selected curves")
        self.btn_chk_all.setToolTip("Check all curves on the list")
        self.btn_hide.setToolTip("Clear chart, uncheck all selected curves")
        self.btn_reset.setToolTip("Reset to original dataset, remove all modified curves")

    # ---------------------- Create menus ----------------------
    def create_menus(self):
        self.save_menu = QMenu(self)
        self.action_save_png = self.save_menu.addAction("📈 Save chart (.png)")
        self.action_export_csv = self.save_menu.addAction("📆 Export data (.csv)")
        self.btn_save.setMenu(self.save_menu)

        self.normalize_menu = QMenu(self)
        self.action_norm_IU = self.normalize_menu.addAction("🔨 Normalize by I/U")
        self.action_norm_Imax = self.normalize_menu.addAction("🔧 Normalize by I\u2098\u2090\u2093")
        self.btn_normalize.setMenu(self.normalize_menu)

        self.filter_menu = QMenu(self)
        self.action_savgol = self.filter_menu.addAction("Savitzky-Golay")
        self.action_wiener = self.filter_menu.addAction("Wiener")
        self.btn_filter.setMenu(self.filter_menu)

    # -------------------- Style buttons -------------------
    def style_buttons(self):
        all_buttons = [
            self.btn_save,
            self.btn_normalize,
            self.btn_filter,
            self.btn_derivative,
            self.btn_subtract,
            self.btn_divide,
            self.btn_average,
            self.btn_fit,
        ]
        Style.style_buttons(all_buttons)
        Style.style_checkbox(self.chk_swap_order)
        self.chk_swap_order.setFixedWidth(76)

    # ---------------------- Layout ----------------------
    def create_layout(self):
        layout, self.toolbar_frame = Style.create_toolbar_frame()
        # Adding buttons to toolbar
        for item in [self.btn_save, Style.create_pixel_divider(), self.btn_normalize,
                     self.btn_filter, Style.create_pixel_divider(), self.btn_subtract,
                     self.btn_divide, self.chk_swap_order, Style.create_pixel_divider(),
                     self.btn_average, self.btn_derivative, Style.create_pixel_divider(),
                     self.btn_fit
                     ]:
            layout.addWidget(item)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_layout.addWidget(self.toolbar_frame)
        main_layout.addStretch(1)

        # ---------------------- Menu styling ----------------------
    def style_menu(self):
        menu_list = [self.save_menu,
                     self.normalize_menu,
                     self.filter_menu]
        for menu in menu_list:
            Style.style_menu(menu)
