import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from regression import linear_regression, parabolic_regression, sixth_deg_regression

class PlotCanvas(FigureCanvas):
    """A Matplotlib canvas embedded in a QWidget."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_data(self, x, y, title="Sample Data"):
        """Plot (x, y) on the canvas."""
        self.axes.clear()
        self.axes.scatter(x, y, s=10, c="blue", alpha=0.7, label="Generated Data")
        self.axes.set_title(title)
        self.axes.legend()
        self.draw()

    def plot_loss(self, loss_array, title="Loss Over Iterations"):
        """Plot the loss array on the canvas."""
        self.axes.clear()
        self.axes.plot(loss_array, label="Loss")
        self.axes.set_xlabel("Iteration")
        self.axes.set_ylabel("Loss")
        self.axes.set_title(title)
        self.axes.legend()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gradient Descent Demo")

        # Data holders
        self.x_data = None
        self.y_data = None

        # ---------------------------
        #   Widgets / Layout
        # ---------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Input fields for generating parabolic sample
        input_layout = QGridLayout()

        input_layout.addWidget(QLabel("Parabolic coefficient a:"), 0, 0)
        self.a_edit = QLineEdit("1.0")
        input_layout.addWidget(self.a_edit, 0, 1)

        input_layout.addWidget(QLabel("Parabolic coefficient b:"), 1, 0)
        self.b_edit = QLineEdit("0.5")
        input_layout.addWidget(self.b_edit, 1, 1)

        input_layout.addWidget(QLabel("Parabolic coefficient c:"), 2, 0)
        self.c_edit = QLineEdit("0.2")
        input_layout.addWidget(self.c_edit, 2, 1)

        input_layout.addWidget(QLabel("Noise variance:"), 3, 0)
        self.var_edit = QLineEdit("0.5")
        input_layout.addWidget(self.var_edit, 3, 1)

        self.generate_btn = QPushButton("Generate Sample")
        self.generate_btn.clicked.connect(self.generate_sample)
        input_layout.addWidget(self.generate_btn, 4, 0, 1, 2)

        layout.addLayout(input_layout)

        # Plot canvas for data
        self.data_canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.data_canvas)

        # Model selection and gradient descent input
        gd_layout = QHBoxLayout()

        gd_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Linear", "Parabolic", "6th Degree"])
        gd_layout.addWidget(self.model_combo)

        gd_layout.addWidget(QLabel("Iterations:"))
        self.iter_edit = QLineEdit("100")
        gd_layout.addWidget(self.iter_edit)

        gd_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_edit = QLineEdit("0.001")
        gd_layout.addWidget(self.lr_edit)

        self.train_btn = QPushButton("Run Gradient Descent")
        self.train_btn.clicked.connect(self.run_gradient_descent)
        gd_layout.addWidget(self.train_btn)

        layout.addLayout(gd_layout)

        # Plot canvas for loss
        self.loss_canvas = PlotCanvas(self, width=5, height=3)
        layout.addWidget(self.loss_canvas)

    def generate_sample(self):
        """Generate (x, y) data from a + b*x + c*x^2 + noise."""
        try:
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            c = float(self.c_edit.text())
            variance = float(self.var_edit.text())

            # Generate sample
            np.random.seed(0)  # for reproducibility
            self.x_data = np.linspace(-5, 5, 50)
            noise = np.random.normal(0, variance, self.x_data.shape)
            self.y_data = a + b*self.x_data + c*(self.x_data**2) + noise

            # Plot the generated sample
            self.data_canvas.plot_data(self.x_data, self.y_data, title="Generated Parabolic Sample")
        except ValueError:
            pass  # In a real app you'd show an error message

    def run_gradient_descent(self):
        """Run gradient descent on the selected model using the generated data."""
        if self.x_data is None or self.y_data is None:
            return  # No data to train on

        try:
            iterations = int(self.iter_edit.text())
            lr = float(self.lr_edit.text())
            model_name = self.model_combo.currentText()

            x = self.x_data
            y = self.y_data

            # Pick initial params depending on model
            if model_name == "Linear":
                # (a, b)
                params = [0.0, 0.0]
                loss_array, final_params = linear_regression(x, y, params, iterations, lr)

            elif model_name == "Parabolic":
                # (a, b, c)
                params = [0.0, 0.0, 0.0]
                loss_array, final_params = parabolic_regression(x, y, params, iterations, lr)

            else:
                # 6th Degree
                # (a, b, c, d, e, f, g)
                params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                loss_array, final_params = sixth_deg_regression(x, y, params, iterations, lr)

            # Plot the loss
            self.loss_canvas.plot_loss(loss_array, title=f"{model_name} Loss Over Iterations")

        except ValueError:
            pass  # In a real app you'd show an error message


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
