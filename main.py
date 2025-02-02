import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


import scipy.stats as stats



from regression import linear_regression, parabolic_regression, sixth_deg_regression


class PlotCanvas:
    """
    class that provides plotting generated sample, loss function values and regression curve
    """

    def __init__(self, parent, width=7, height=5, dpi=100) -> None:
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

    def plot_data(self, x: list[float], y: list[float], title: str = "Вибірка") -> None:
        """
        function that plots generated sample
        """

        self.axes.clear()

        self.axes.scatter(x, y, s=10, c="blue", alpha=0.7)
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.axes.grid()
        self.canvas.draw()

    def plot_loss(self, loss_array: list[float]) -> None:
        """
        function that plots loss function values
        """

        self.axes.clear()
        self.axes.plot(loss_array, label="Loss", color="red")
        self.axes.set_xlabel("Ітерації")
        self.axes.set_ylabel("Функція втрат")
        self.figure.tight_layout()
        self.axes.grid()
        self.canvas.draw()

    def plot_curve(self, x: list[float], y: list[float]) -> None:
        """
        function that plots regression curve
        """

        self.axes.plot(x, y, c="red", alpha=0.7)
        self.figure.tight_layout()
        self.canvas.draw()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Градієнтний спуск")
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

        input_frame = tk.LabelFrame(controls_frame, text="Параметри для генерування вибірки")
        input_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=0)

        tk.Label(input_frame, text="Ступінь поліному:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.gen_degree_var = tk.StringVar(value="Лінійний")
        degree_options = ["Лінійний", "Параболічний", "6-го порядку"]
        self.degree_combo = ttk.Combobox(input_frame, textvariable=self.gen_degree_var,
                                         values=degree_options, state="readonly", width=12)
        self.degree_combo.grid(row=0, column=1, padx=2, pady=2)
        self.degree_combo.bind("<<ComboboxSelected>>", self.update_param_fields)

        self.param_frame = tk.Frame(input_frame)
        self.param_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        self.param_entries = {}
        self.update_param_fields()

        tk.Label(input_frame, text="СКВ:").grid(row=10, column=0, sticky=tk.W, padx=2, pady=2)
        self.std_var = tk.StringVar(value="10")
        tk.Entry(input_frame, textvariable=self.std_var, width=8).grid(row=10, column=1, padx=2, pady=2)

        tk.Label(input_frame, text="Кількість даних:").grid(row=11, column=0, sticky=tk.W, padx=2, pady=2)
        self.volume_var = tk.StringVar(value="100")
        tk.Entry(input_frame, textvariable=self.volume_var, width=8).grid(row=11, column=1, padx=2, pady=2)

        gen_button = tk.Button(input_frame, text="Згенерувати вибірку", command=self.generate_sample)
        gen_button.grid(row=12, column=0, columnspan=2, pady=4)

        standard_button = tk.Button(input_frame, text="Стандартизувати", command=self.standardization)
        standard_button.grid(row=12, column=3, columnspan=2, pady=4)

        gd_frame = tk.LabelFrame(controls_frame, text="Параметри для градієнтного спуску")
        gd_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=0)

        tk.Label(gd_frame, text="Модель:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.model_var = tk.StringVar(value="Лінійна")
        model_combo = ttk.Combobox(gd_frame, textvariable=self.model_var,
                                   values=["Лінійна", "Параболічна", "6го порядку"],
                                   width=12, state="readonly")
        model_combo.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(gd_frame, text="Кількість ітерацій:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.iter_var = tk.StringVar(value="100")
        tk.Entry(gd_frame, textvariable=self.iter_var, width=6).grid(row=1, column=1, padx=2, pady=2)

        tk.Label(gd_frame, text="Швидкість навчання:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.lr_var = tk.StringVar(value="0.001")
        tk.Entry(gd_frame, textvariable=self.lr_var, width=6).grid(row=2, column=1, padx=2, pady=2)

        run_button = tk.Button(gd_frame, text="Градієнтний спуск", command=self.run_gradient_descent)
        run_button.grid(row=3, column=0, columnspan=2, pady=4)

        plots_frame = tk.Frame(self)
        plots_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        plots_frame.grid_rowconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(1, weight=1)

        self.data_frame = tk.Frame(plots_frame, width=750, height=600)
        self.data_frame.grid(row=0, column=0, padx=(0, 5), pady=0)
        self.data_frame.grid_propagate(False)
        self.data_canvas = PlotCanvas(self.data_frame, width=8, height=6)

        self.loss_frame = tk.Frame(plots_frame, width=750, height=600)
        self.loss_frame.grid(row=0, column=1, padx=(5, 0), pady=0)
        self.loss_frame.grid_propagate(False)
        self.loss_canvas = PlotCanvas(self.loss_frame, width=8, height=6)

        total_width = 750 * 2 + 40
        total_height = 500 + 200 + 40
        self.geometry(f"{total_width}x{total_height}")

    def update_param_fields(self, event=None) -> None:
        """
        function that updates parameter input fields according to user input
        """

        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        degree = self.gen_degree_var.get()
        if degree == "Лінійний":
            param_names = ["a", "b"]
        elif degree == "Параболічний":
            param_names = ["a", "b", "c"]
        elif degree == "6-го порядку":
            param_names = ["a", "b", "c", "d", "e", "f", "g"]
        else:
            param_names = []

        for i, name in enumerate(param_names):
            label = tk.Label(self.param_frame, text=f"{name}:")
            label.grid(row=0, column=i * 2, sticky=tk.E, padx=2, pady=2)

            var = tk.StringVar(value=f"{i}")
            entry = tk.Entry(self.param_frame, textvariable=var, width=6)
            entry.grid(row=0, column=i * 2 + 1, sticky=tk.W, padx=2, pady=2)
            self.param_entries[name] = var

    def standardization(self) -> None:
        """
        function that standardizes input data
        """
        x_mean = np.mean(self.x_data)
        y_mean = np.mean(self.y_data)

        x_std = np.std(self.x_data)
        y_std = np.std(self.y_data)

        self.x_data = (self.x_data - x_mean) / x_std
        self.y_data = (self.y_data - y_mean) / y_std

        self.data_canvas.plot_data(self.x_data, self.y_data, title="Стандартизована вибірка")

    def generate_sample(self) -> None:
        """Generate (x, y) data using the selected polynomial degree and corresponding parameters."""

        try:
            degree = self.gen_degree_var.get()
            params = [float(self.param_entries[key].get()) for key in sorted(self.param_entries.keys())]

            std = float(self.std_var.get())
            volume = int(self.volume_var.get())

            np.random.seed(42)
            self.x_data = np.linspace(-10, 10, volume)
            noise = np.random.normal(0, std, self.x_data.shape)

            if degree == "Лінійний":
                a, b = params
                self.y_data = a + b * self.x_data + noise

            elif degree == "Параболічний":
                a, b, c = params
                self.y_data = a + b * self.x_data + c * self.x_data ** 2 + noise

            elif degree == "6-го порядку":
                if len(params) != 7:
                    raise ValueError("Для 6-го порядку потрібно ввести 7 коефіцієнтів.")
                a, b, c, d, e, f, g = params
                self.y_data = (a + b * self.x_data + c * self.x_data ** 2 +
                               d * self.x_data ** 3 + e * self.x_data ** 4 +
                               f * self.x_data ** 5 + g * self.x_data ** 6 + noise)
            else:
                raise ValueError("Невідомий ступінь поліному для генерування вибірки.")

            self.data_canvas.plot_data(self.x_data, self.y_data, title="Згенерована вибірка")

        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Будь ласка, введіть коректні числові значення для параметрів, СКВ і кількості даних.")

    def run_gradient_descent(self) -> None:
        """Run gradient descent on the selected model using the generated data."""

        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("No Data", "Будь ласка, згенеруйте вибірку спочатку.")
            return

        try:
            iterations = int(self.iter_var.get())
            lr = float(self.lr_var.get())
            model_name = self.model_var.get()

            x = self.x_data
            y = self.y_data

            np.random.seed(42)

            if model_name == "Лінійна":
                params = [np.random.rand() - 0.5, np.random.rand() - 0.5]
                loss_array, final_params = linear_regression(x, y, params, iterations, lr)

            elif model_name == "Параболічна":
                params = [np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5]
                loss_array, final_params = parabolic_regression(x, y, params, iterations, lr)

            elif model_name == "6го порядку":
                params = [np.random.rand() - 0.5 for _ in range(7)]
                loss_array, final_params = sixth_deg_regression(x, y, params, iterations, lr)

            else:
                raise ValueError("Невідома модель.")

            self.loss_canvas.plot_loss(loss_array)
            # self.show_final_params_window(final_params, loss_array[-1])
            self.show_final_params_window(final_params)

            """plot regression curve"""
            if model_name == "Лінійна":
                a, b = final_params
                y_pred = [a + b * x for x in self.x_data]
                self.data_canvas.plot_curve(x, y_pred)


            elif model_name == "Параболічна":
                a, b, c = final_params
                y_pred = [a + b * x + c * x ** 2 for x in self.x_data]

                self.data_canvas.plot_curve(x, y_pred)

            elif model_name == "6го порядку":
                a, b, c, d, e, f, g = final_params
                y_pred = [a + b * x + c * x ** 2 + d * x ** 3 + a + b * x ** 4 + c * x ** 5 + d * x ** 6 for x in
                          self.x_data]

                self.data_canvas.plot_curve(x, y_pred)





        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Будь ласка, введіть коректні числові значення для ітерацій та швидкості навчання.")


    """****************************************************************************************************************"""
    def compute_confidence_intervals(self, x, y, final_params, alpha=0.05):
        """
        Compute confidence intervals for regression parameters.

        Args:
            x (np.array): Feature values.
            y (np.array): Target values.
            final_params (np.array): Estimated parameters.
            alpha (float): Significance level (default=0.05 for 95% CI).

        Returns:
            dict: Confidence intervals for each parameter.
        """
        x = np.array(x)
        y = np.array(y)
        final_params = np.array(final_params, dtype=np.float64)
        n = len(x)
        k = len(final_params)  # Number of parameters

        # Compute predictions
        if k == 2:  # Linear regression: y = a + bx
            y_pred = final_params[0] + final_params[1] * x
            X = np.column_stack((np.ones(n), x))
        elif k == 3:  # Parabolic regression: y = a + bx + cx^2
            y_pred = final_params[0] + final_params[1] * x + final_params[2] * x ** 2
            X = np.column_stack((np.ones(n), x, x ** 2))
        elif k == 7:  # 6th-degree polynomial regression
            y_pred = (final_params[0] + final_params[1] * x + final_params[2] * x ** 2 +
                      final_params[3] * x ** 3 + final_params[4] * x ** 4 +
                      final_params[5] * x ** 5 + final_params[6] * x ** 6)
            X = np.column_stack((np.ones(n), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6))
        else:
            raise ValueError("Unsupported model type")

        # Compute residual variance (sigma^2)
        residuals = y - y_pred
        sigma_squared = np.sum(residuals ** 2) / (n - k)
        sigma = np.sqrt(sigma_squared)

        # Compute (X^T * X)^(-1)
        XtX_inv = np.linalg.inv(X.T @ X)

        # Compute standard errors of parameters
        SE_params = np.sqrt(sigma_squared * np.diag(XtX_inv))

        # Determine t-critical value dynamically
        t_value = stats.t.ppf(1 - alpha / 2, df=n - k)

        # Compute confidence intervals
        conf_intervals = {}
        for i, (param, se) in enumerate(zip(final_params, SE_params)):
            lower_bound = param - t_value * se
            upper_bound = param + t_value * se
            conf_intervals[i] = (lower_bound, upper_bound)

        return conf_intervals

    def show_final_params_window(self, final_params: list[float]):
        """Create a pop-up (Toplevel) window to display final model parameters and confidence intervals."""
        pop = tk.Toplevel(self)
        pop.title("Final Parameters & Confidence Intervals")
        pop.resizable(False, False)

        n = len(self.x_data)
        model_name = self.model_var.get()

        # Compute confidence intervals
        conf_intervals = self.compute_confidence_intervals(self.x_data, self.y_data, final_params)

        # Define parameter names
        if model_name == "Лінійна":
            param_names = ['a', 'b']
        elif model_name == "Параболічна":
            param_names = ['a', 'b', 'c']
        elif model_name == "6го порядку":
            param_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        else:
            param_names = []

        # Format text
        param_text = "Оцінка параметрів:\n"
        for i, param in enumerate(final_params):
            param_text += f"{param_names[i]} = {param:.4f}\n"

        param_text += "\nІнтервальна оцінка параметрів (95% CI):\n"
        for i, (lower, upper) in conf_intervals.items():
            param_text += f"{param_names[i]}:  [{lower:.4f}, {upper:.4f}]\n"

        lbl = tk.Label(pop, text=param_text, padx=20, pady=20, justify=tk.LEFT)
        lbl.pack()

        # Center the pop-up window
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





    """****************************************************************************************************************"""



    # def show_final_params_window(self, final_params: list[float], loss: float) -> None:
    #     pop = tk.Toplevel(self)
    #     pop.resizable(False, False)
    #
    #     param_text = "Оцінка параметрів: " + ", ".join(f"{p:.4f}" for p in final_params)
    #
    #     """***********************************************************************************************************"""
    #     """Interval estimation of parameters & checking hypothesis"""
    #     n = len(self.x_data)
    #
    #     # t_quant = 0
    #     if 10 < n < 120:
    #         t_quant = 2.105
    #     else:
    #         t_quant = 1.96
    #
    #     param_text += '\n\n Інтервальна оцінка параметрів:\n'
    #
    #     model_name = self.model_var.get()
    #
    #     if model_name == "Лінійна":
    #         abc = ['a', 'b']
    #         for i in range(len(final_params)):
    #
    #             param_text += f'{abc[i]}:  [{final_params[i] - t_quant * loss},  {final_params[i] + t_quant * loss}]\n'
    #
    #
    #
    #     elif model_name == "Параболічна":
    #         abc = ['a', 'b', 'c']
    #         for i in range(len(final_params)):
    #             param_text += f'{abc[i]}:  [{final_params[i] - t_quant * loss},  {final_params[i] + t_quant * loss}\n'
    #
    #
    #
    #     elif model_name == "6го порядку":
    #         abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    #         for i in range(len(final_params)):
    #             param_text += f'{abc[i]}:  [{final_params[i] - t_quant * loss},  {final_params[i] + t_quant * loss}\n'
    #
    #
    #     """***********************************************************************************************************"""
    #
    #     lbl = tk.Label(pop, text=param_text, padx=20, pady=20)
    #     lbl.pack()
    #
    #     self.update_idletasks()
    #     pop_width = pop.winfo_reqwidth()
    #     pop_height = pop.winfo_reqheight()
    #     main_x = self.winfo_x()
    #     main_y = self.winfo_y()
    #     main_width = self.winfo_width()
    #     main_height = self.winfo_height()
    #     pos_x = main_x + (main_width // 2) - (pop_width // 2)
    #     pos_y = main_y + (main_height // 2) - (pop_height // 2)
    #     pop.geometry(f"+{pos_x}+{pos_y}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
