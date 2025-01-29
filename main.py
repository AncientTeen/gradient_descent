import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from regression import linear_regression, parabolic_regression, sixth_deg_regression

class PlotCanvas:
    """
    A wrapper around a Matplotlib Figure for embedding in a Tkinter widget.
    Provides methods to plot data and loss curves.
    """

    def __init__(self, parent, width=7, height=5, dpi=100):
        self.parent = parent
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.parent)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

    def plot_data(self, x, y, title="Sample Data"):
        self.axes.clear()
        self.axes.scatter(x, y, s=10, c="blue", alpha=0.7, label="Generated Data")
        self.axes.set_title(title)
        self.axes.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_loss(self, loss_array, title="Loss Over Iterations"):
        self.axes.clear()
        self.axes.plot(loss_array, label="Loss", color="red")
        self.axes.set_xlabel("Iteration")
        self.axes.set_ylabel("Loss")
        self.axes.set_title(title)
        self.axes.legend()
        self.figure.tight_layout()
        self.canvas.draw()

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gradient Descent Demo (Tkinter)")
        self.resizable(False, False)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.x_data = None
        self.y_data = None


        controls_frame = tk.Frame(self)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        controls_frame.grid_rowconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)


        input_frame = tk.LabelFrame(controls_frame, text="Data Parameters")
        input_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=0)

        tk.Label(input_frame, text="Coefficient a:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.a_var = tk.StringVar(value="1.0")
        tk.Entry(input_frame, textvariable=self.a_var, width=8).grid(row=0, column=1, padx=2, pady=2)

        tk.Label(input_frame, text="Coefficient b:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.b_var = tk.StringVar(value="0.5")
        tk.Entry(input_frame, textvariable=self.b_var, width=8).grid(row=1, column=1, padx=2, pady=2)

        tk.Label(input_frame, text="Coefficient c:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.c_var = tk.StringVar(value="0.2")
        tk.Entry(input_frame, textvariable=self.c_var, width=8).grid(row=2, column=1, padx=2, pady=2)

        tk.Label(input_frame, text="Noise Variance:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.var_var = tk.StringVar(value="0.5")
        tk.Entry(input_frame, textvariable=self.var_var, width=8).grid(row=3, column=1, padx=2, pady=2)

        gen_button = tk.Button(input_frame, text="Generate Sample", command=self.generate_sample)
        gen_button.grid(row=4, column=0, columnspan=2, pady=4)


        gd_frame = tk.LabelFrame(controls_frame, text="Gradient Descent Controls")
        gd_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=0)

        tk.Label(gd_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.model_var = tk.StringVar(value="Linear")
        model_combo = ttk.Combobox(
            gd_frame,
            textvariable=self.model_var,
            values=["Linear", "Parabolic", "6th Degree"],
            width=12,
            state="readonly"
        )
        model_combo.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(gd_frame, text="Iterations:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.iter_var = tk.StringVar(value="100")
        tk.Entry(gd_frame, textvariable=self.iter_var, width=6).grid(row=1, column=1, padx=2, pady=2)

        tk.Label(gd_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.lr_var = tk.StringVar(value="0.001")
        tk.Entry(gd_frame, textvariable=self.lr_var, width=6).grid(row=2, column=1, padx=2, pady=2)

        run_button = tk.Button(gd_frame, text="Run Gradient Descent", command=self.run_gradient_descent)
        run_button.grid(row=3, column=0, columnspan=2, pady=4)


        plots_frame = tk.Frame(self)
        plots_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0,5))

        plots_frame.grid_rowconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(1, weight=1)

        self.data_frame = tk.Frame(plots_frame, width=700, height=500)
        self.data_frame.grid(row=0, column=0, padx=(0,5), pady=0)
        self.data_frame.grid_propagate(False)

        self.data_canvas = PlotCanvas(self.data_frame, width=7, height=5)

        # Loss Plot (Right)
        self.loss_frame = tk.Frame(plots_frame, width=700, height=500)
        self.loss_frame.grid(row=0, column=1, padx=(5,0), pady=0)
        self.loss_frame.grid_propagate(False)

        self.loss_canvas = PlotCanvas(self.loss_frame, width=7, height=5)


        total_width = 700 * 2 + 40
        total_height = 500 + 200 + 40

        self.geometry(f"{total_width}x{total_height}")

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
            # Show an error message to the user
            messagebox.showerror("Invalid Input", "Please enter valid numerical values for parameters and variance.")

    def run_gradient_descent(self):
        """Run gradient descent on the selected model using the generated data."""
        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("No Data", "Please generate sample data first.")
            return

        try:
            iterations = int(self.iter_var.get())
            lr = float(self.lr_var.get())
            model_name = self.model_var.get()

            x = self.x_data
            y = self.y_data

            if model_name == "Linear":

                params = [np.random.rand() - 0.5, np.random.rand() - 0.5]
                print(params)
                loss_array, final_params = linear_regression(x, y, params, iterations, lr)

            elif model_name == "Parabolic":
                params = [np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5]
                print(params)

                loss_array, final_params = parabolic_regression(x, y, params, iterations, lr)

            else:
                params = [np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5]
                print(params)

                loss_array, final_params = sixth_deg_regression(x, y, params, iterations, lr)

            self.loss_canvas.plot_loss(loss_array, title=f"{model_name} Loss Over Iterations")

            self.show_final_params_window(final_params)

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numerical values for iterations and learning rate.")

    def show_final_params_window(self, final_params):
        pop = tk.Toplevel(self)
        pop.title("Final Parameters")
        pop.resizable(False, False)

        param_text = "Final parameters: " + ", ".join(f"{p:.4f}" for p in final_params)
        lbl = tk.Label(pop, text=param_text, padx=20, pady=20)
        lbl.pack()

        self.update_idletasks()
        pop_width = pop.winfo_reqwidth()
        pop_height = pop.winfo_reqheight()
        main_x = self.winfo_x()
        main_y = self.winfo_y()
        main_width = self.winfo_width()
        main_height = self.winfo_height()
        pos_x = main_x + (main_width // 2) - (pop_width // 2)
        pos_y = main_y + (main_height // 2) - (pop_height // 2)
        pop.geometry(f"+{pos_x}+{pos_y}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
