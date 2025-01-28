import tkinter as tk
from tkinter import ttk
import numpy as np

# If your regression functions are in a separate file called `regression.py`:
from regression import linear_regression, parabolic_regression, sixth_deg_regression
# from regression import linear_regression, sixth_deg_regression

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg





class PlotCanvas:
    """
    A wrapper around a Matplotlib Figure for embedding in a Tkinter widget.
    Provides methods to plot data and loss curves.
    """
    def __init__(self, parent, width=5, height=4, dpi=100):
        self.parent = parent
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)

        # Create canvas and add it to the parent
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_data(self, x, y, title="Sample Data"):
        """Plot (x, y) on the canvas."""
        self.axes.clear()
        self.axes.scatter(x, y, s=10, c="blue", alpha=0.7, label="Generated Data")
        self.axes.set_title(title)
        self.axes.legend()
        self.canvas.draw()

    def plot_loss(self, loss_array, title="Loss Over Iterations"):
        """Plot the loss array on the canvas."""
        self.axes.clear()
        self.axes.plot(loss_array, label="Loss")
        self.axes.set_xlabel("Iteration")
        self.axes.set_ylabel("Loss")
        self.axes.set_title(title)
        self.axes.legend()
        self.canvas.draw()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gradient Descent Demo (Tkinter)")
        self.geometry("900x700")

        # Data holders
        self.x_data = None
        self.y_data = None

        # ---------------------------
        #   Top Frame for Inputs
        # ---------------------------
        input_frame = tk.Frame(self)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Parabolic coefficient a
        tk.Label(input_frame, text="Parabolic coefficient a:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.a_var = tk.StringVar(value="1.0")
        tk.Entry(input_frame, textvariable=self.a_var, width=8).grid(row=0, column=1, padx=2, pady=2)

        # Parabolic coefficient b
        tk.Label(input_frame, text="Parabolic coefficient b:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.b_var = tk.StringVar(value="0.5")
        tk.Entry(input_frame, textvariable=self.b_var, width=8).grid(row=1, column=1, padx=2, pady=2)

        # Parabolic coefficient c
        tk.Label(input_frame, text="Parabolic coefficient c:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.c_var = tk.StringVar(value="0.2")
        tk.Entry(input_frame, textvariable=self.c_var, width=8).grid(row=2, column=1, padx=2, pady=2)

        # Noise variance
        tk.Label(input_frame, text="Noise variance:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.var_var = tk.StringVar(value="0.5")
        tk.Entry(input_frame, textvariable=self.var_var, width=8).grid(row=3, column=1, padx=2, pady=2)

        # Generate sample button
        gen_button = tk.Button(input_frame, text="Generate Sample", command=self.generate_sample)
        gen_button.grid(row=4, column=0, columnspan=2, pady=4)

        # ---------------------------
        #   Canvas for data display
        # ---------------------------
        self.data_frame = tk.Frame(self)
        self.data_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.data_canvas = PlotCanvas(self.data_frame, width=5, height=3)

        # ---------------------------
        #   Middle Frame for GD Input
        # ---------------------------
        gd_frame = tk.Frame(self)
        gd_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Model selection
        tk.Label(gd_frame, text="Model:").pack(side=tk.LEFT, padx=2)
        self.model_var = tk.StringVar(value="Linear")
        model_combo = ttk.Combobox(gd_frame, textvariable=self.model_var, values=["Linear", "Parabolic", "6th Degree"], width=12)
        model_combo.pack(side=tk.LEFT, padx=2)

        # Iterations
        tk.Label(gd_frame, text="Iterations:").pack(side=tk.LEFT, padx=2)
        self.iter_var = tk.StringVar(value="100")
        tk.Entry(gd_frame, textvariable=self.iter_var, width=6).pack(side=tk.LEFT, padx=2)

        # Learning rate
        tk.Label(gd_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=2)
        self.lr_var = tk.StringVar(value="0.001")
        tk.Entry(gd_frame, textvariable=self.lr_var, width=6).pack(side=tk.LEFT, padx=2)

        # Run Gradient Descent button
        tk.Button(gd_frame, text="Run Gradient Descent", command=self.run_gradient_descent)\
            .pack(side=tk.LEFT, padx=10)

        # ---------------------------
        #   Canvas for loss display
        # ---------------------------
        self.loss_frame = tk.Frame(self)
        self.loss_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.loss_canvas = PlotCanvas(self.loss_frame, width=5, height=3)

    def generate_sample(self):
        """Generate (x, y) data from a + b*x + c*x^2 + noise."""
        try:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            c = float(self.c_var.get())
            variance = float(self.var_var.get())

            np.random.seed(0)  # for reproducibility
            self.x_data = np.linspace(-5, 5, 50)
            noise = np.random.normal(0, variance, self.x_data.shape)
            self.y_data = a + b*self.x_data + c*self.x_data**2 + noise

            # Plot the generated sample
            self.data_canvas.plot_data(self.x_data, self.y_data, title="Generated Parabolic Sample")

        except ValueError:
            # In a production app, you might show a warning dialog here.
            pass

    def run_gradient_descent(self):
        """Run gradient descent on the selected model using the generated data."""
        if self.x_data is None or self.y_data is None:
            return  # No data to train on, or not generated yet.

        try:
            iterations = int(self.iter_var.get())
            lr = float(self.lr_var.get())
            model_name = self.model_var.get()

            x = self.x_data
            y = self.y_data

            # Pick initial params depending on model
            if model_name == "Linear":
                # (a, b)
                params = [0.5, 0.5]
                loss_array, final_params = linear_regression(x, y, params, iterations, lr)

            elif model_name == "Parabolic":
                # (a, b, c)
                params = [0.5, 0.5, 0.5]
                loss_array, final_params = parabolic_regression(x, y, params, iterations, lr)

            else:
                # 6th Degree
                # (a, b, c, d, e, f, g)
                params = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                loss_array, final_params = sixth_deg_regression(x, y, params, iterations, lr)

            # Plot the loss
            self.loss_canvas.plot_loss(loss_array, title=f"{model_name} Loss Over Iterations")

        except ValueError:
            # In a production app, you might show a warning dialog here.
            pass


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()


