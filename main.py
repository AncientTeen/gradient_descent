import numpy as np
import tkinter as tk
from math import comb
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



from regression import linear_regression, parabolic_regression, sixth_deg_regression


class PlotCanvas:
    """
    class provides plotting generated sample, loss function values and regression curve
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

        self.x_mean = None
        self.y_mean = None
        self.x_std = None
        self.y_std = None

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

        tk.Label(input_frame, text="Зсув по Х:").grid(row=12, column=0, sticky=tk.W, padx=2, pady=2)
        self.shift_x = tk.StringVar(value="0")
        tk.Entry(input_frame, textvariable=self.shift_x, width=8).grid(row=12, column=1, padx=2, pady=2)

        tk.Label(input_frame, text="Зсув по Y:").grid(row=12, column=2, sticky=tk.W, padx=2, pady=2)
        self.shift_y = tk.StringVar(value="0")
        tk.Entry(input_frame, textvariable=self.shift_y, width=8).grid(row=12, column=3, padx=2, pady=2)

        gen_button = tk.Button(input_frame, text="Згенерувати вибірку", command=self.generate_sample)
        gen_button.grid(row=13, column=0, columnspan=2, pady=4)

        standard_button = tk.Button(input_frame, text="Стандартизувати", command=self.standardization)
        standard_button.grid(row=13, column=3, columnspan=2, pady=4)

        gd_frame = tk.LabelFrame(controls_frame, text="Параметри для градієнтного спуску")
        gd_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=0)

        tk.Label(gd_frame, text="Модель:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.model_var = tk.StringVar(value="Лінійна")
        model_combo = ttk.Combobox(gd_frame, textvariable=self.model_var,
                                   values=["Лінійна", "Параболічна", "6-го порядку"],
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
        function updates parameter input fields according to user input
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
        function standardizes input data
        """
        self.x_mean = np.mean(self.x_data)
        self.y_mean = np.mean(self.y_data)

        self.x_std = np.std(self.x_data)
        self.y_std = np.std(self.y_data)

        self.x_data = (self.x_data - self.x_mean) / self.x_std
        self.y_data = (self.y_data - self.y_mean) / self.y_std

        self.data_canvas.plot_data(self.x_data, self.y_data, title="Стандартизована вибірка")

    def generate_sample(self) -> None:
        """
        generate (x, y) data using the selected polynomial degree and corresponding parameters
        """

        try:
            degree = self.gen_degree_var.get()
            params = [float(self.param_entries[key].get()) for key in sorted(self.param_entries.keys())]

            std = float(self.std_var.get())
            volume = int(self.volume_var.get())

            shift_x = float(self.shift_x.get())
            shift_y = float(self.shift_y.get())

            np.random.seed(42)
            self.x_data = np.linspace(-10, 10, volume)
            noise = np.random.normal(0, std, self.x_data.shape)

            if degree == "Лінійний":
                a, b = params
                self.y_data = a + b * (self.x_data + shift_x) + noise + shift_y

            elif degree == "Параболічний":
                a, b, c = params
                self.y_data = a + b * (self.x_data + shift_x) + c * (self.x_data + shift_x) ** 2 + noise + shift_y

            elif degree == "6-го порядку":

                a, b, c, d, e, f, g = params
                self.y_data = (a + b * (self.x_data + shift_x) + c * (self.x_data + shift_x) ** 2 +
                               d * (self.x_data + shift_x) ** 3 + e * (self.x_data + shift_x) ** 4 +
                               f * (self.x_data + shift_x) ** 5 + g * (self.x_data + shift_x) ** 6 + noise)  + shift_y
            else:
                raise ValueError("Невідомий ступінь поліному для генерування вибірки.")

            self.data_canvas.plot_data(self.x_data, self.y_data, title="Згенерована вибірка")

        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Будь ласка, введіть коректні числові значення для параметрів, СКВ і кількості даних.")


    def run_gradient_descent(self):
        """
        function runs gradient descent
         - previously standardized data and then return it to origin
        """

        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("No Data", "Будь ласка, згенеруйте вибірку спочатку.")
            return

        try:

            iterations = int(self.iter_var.get())
            lr = float(self.lr_var.get())
            model_name = self.model_var.get()

            x_arr = np.array(self.x_data, dtype=float)
            y_arr = np.array(self.y_data, dtype=float)

            self.x_mean = x_arr.mean()
            self.x_std = x_arr.std()
            self.y_mean = y_arr.mean()
            self.y_std = y_arr.std()

            if abs(self.x_std) < 1e-14 or abs(self.y_std) < 1e-14:
                messagebox.showerror("Error", "Дані майже незмінні; неможливо стандартизувати.")
                return

            x_std = (x_arr - self.x_mean) / self.x_std
            y_std = (y_arr - self.y_mean) / self.y_std

            if model_name == "Лінійна":
                init_params_std = np.random.rand(2) - 0.5
                loss_array, final_params_std = linear_regression(
                    x_std, y_std, init_params_std, iterations, lr, shift_x=0.0
                )
            elif model_name == "Параболічна":
                init_params_std = np.random.rand(3) - 0.5
                loss_array, final_params_std = parabolic_regression(
                    x_std, y_std, init_params_std, iterations, lr, shift_x=0.0
                )
            elif model_name == "6-го порядку":
                init_params_std = np.random.rand(7) - 0.5
                loss_array, final_params_std = sixth_deg_regression(
                    x_std, y_std, init_params_std, iterations, lr, shift_x=0.0
                )
            else:
                messagebox.showerror("Error", "Невідома модель.")
                return

            self.loss_canvas.plot_loss(loss_array)

            x_plot = np.linspace(x_arr.min(), x_arr.max(), 300)
            x_plot_std = (x_plot - self.x_mean) / self.x_std

            if model_name == "Лінійна":
                a_std, b_std = final_params_std
                y_plot_std = a_std + b_std * x_plot_std
            elif model_name == "Параболічна":
                a_std, b_std, c_std = final_params_std
                y_plot_std = a_std + b_std * x_plot_std + c_std * (x_plot_std ** 2)
            else:
                a_s, b_s, c_s, d_s, e_s, f_s, g_s = final_params_std
                y_plot_std = (a_s
                              + b_s * x_plot_std
                              + c_s * (x_plot_std ** 2)
                              + d_s * (x_plot_std ** 3)
                              + e_s * (x_plot_std ** 4)
                              + f_s * (x_plot_std ** 5)
                              + g_s * (x_plot_std ** 6))

            y_plot = self.y_mean + self.y_std * y_plot_std
            self.data_canvas.plot_curve(x_plot, y_plot)

            if model_name == "Лінійна":
                a_std, b_std = final_params_std
                A1 = (self.y_std / self.x_std) * b_std
                A0 = (self.y_mean + self.y_std * a_std) - A1 * self.x_mean
                unstd_params = [A0, A1]

                self.show_final_params_window(unstd_params)

            elif model_name == "Параболічна":
                a_std, b_std, c_std = final_params_std
                A2 = c_std * (self.y_std / (self.x_std ** 2))
                A1 = (b_std * (self.y_std / self.x_std)) - 2 * self.x_mean * A2
                A0 = (self.y_mean
                      + self.y_std * a_std
                      - self.x_mean * (b_std * (self.y_std / self.x_std))
                      + c_std * (self.y_std / (self.x_std ** 2)) * (self.x_mean ** 2))
                unstd_params = [A0, A1, A2]

                self.show_final_params_window(unstd_params)

            elif model_name == "6-го порядку":
                coefs_std = final_params_std
                A = [0.0] * 7
                for k in range(7):
                    if k == 0:
                        factor = self.y_std * coefs_std[k]
                    else:
                        factor = self.y_std * coefs_std[k] / (self.x_std ** k)

                    for m in range(k + 1):
                        A[m] += factor * comb(k, m) * ((-self.x_mean) ** (k - m))

                A[0] += self.y_mean
                self.show_final_params_window(A)
            else:
                self.show_final_params_window(final_params_std)
            # self.show_final_params_window(final_params_std)

        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Будь ласка, введіть коректні числові значення для ітерацій/швидкості навчання.")

    def show_final_params_window(self, final_params: list[float]) -> None:
        """
        function shows final estimation of parameters
        """


        pop = tk.Toplevel(self)
        pop.resizable(False, False)

        param_text = "Оцінка параметрів: \n"
        for p in final_params:
            param_text += f"{p:.3f}\n"

        params = [float(self.param_entries[key].get()) for key in sorted(self.param_entries.keys())]

        """***********************************************************************************************************"""
        param_text += '\n\nВідносна похибка оцінки параметрів:\n'

        model_name = self.model_var.get()

        if model_name == "Лінійна":
            abc = ['a', 'b']
            for i in range(len(final_params)):

                rel_err = (abs(final_params[i] - params[i]) / params[i]) * 100
                param_text += f'{abc[i]}: {rel_err:.4f}%\n'



        elif model_name == "Параболічна":
            abc = ['a', 'b', 'c']
            for i in range(len(final_params)):
                rel_err = (abs(final_params[i] - params[i]) / params[i]) * 100
                param_text += f'{abc[i]}: {rel_err:.4f}%\n'



        elif model_name == "6-го порядку":
            abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            for i in range(len(final_params)):
                rel_err = (abs(final_params[i] - params[i]) / params[i]) * 100
                param_text += f'{abc[i]}: {rel_err:.4f}%\n'


        """***********************************************************************************************************"""

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


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()