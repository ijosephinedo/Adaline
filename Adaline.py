from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

RED = "#fa8174"
YELLOW = "#feffb3"
GREEN = "#b3de69"
BLUE = "#81b1d2"
PINK = "#bc82bd"
ORANGE = "#fdb462"

# Root frame in GUI
root = Tk()
root.title("Adaline")
root.geometry('800x600')
#root.resizable(False, False)
f_inputData = Frame(root)
f_graphs = Frame(root)
f_inputData.pack(side=RIGHT)
f_graphs.pack(side=LEFT, fill=BOTH, expand=1)

# Perceptron variables
eta = DoubleVar()
eta.set(0.1)
maxEpochs = IntVar()
maxEpochs.set(30)
minError = DoubleVar()
minError.set(0.1)
type_pw = IntVar()
type_pw.set(-1)
x = []
d = []
init_w = []
final_w = []
done = False


# Perceptron functions
def init_Weights():
    for i in range(3):
        init_w.append(np.random.uniform(-1, 1))
    hyperplane(init_w, YELLOW, "Initial weigths")
    print("Initial weigths: ")
    print(init_w)
    b_weights['state'] = 'disabled'


def hyperplane(w, color_line, label_name="", style_line='-', line_width=1):
    x = np.arange(-5, 5, 0.01)
    y = (-w[1] * x + w[0]) / w[2]
    if label_name == "":
        ax_p[0].plot(x,
                     y,
                     color=color_line,
                     linestyle=style_line,
                     linewidth=line_width)
    else:
        ax_p[0].plot(x,
                     y,
                     color=color_line,
                     linestyle=style_line,
                     linewidth=line_width,
                     label=label_name)
        ax_p[0].legend()
    fig_p.canvas.draw()


def error_graph(epochs, error_array, color_line):
    x = np.arange(1, epochs + 1)
    max_epochs = "Max epochs: " + str(epochs)
    ax_p[1].plot(x, error_array, color=color_line, label=max_epochs)
    ax_p[1].legend(fontsize=6)
    fig_p.canvas.draw()


def activation_function(v):
    if type_pw.get() == 0:
        return 1 if v >= 0 else 0
    if type_pw.get() == 1:
        return 1 / (1 + np.exp(-v))
    if type_pw.get() == 2:
        return (np.tanh(v) + 1) / 2


def perceptron():
    global done
    global final_w
    error_array = []
    done = False
    epochs = 0
    w = init_w.copy()
    print("Copy of nitial weigths...\n" + str(w))
    print("Training Set...\n" + str(x))
    print("Desired set...\n" + str(d))
    while not done and (epochs < maxEpochs.get()):
        done = True
        epochs += 1
        errors = 0
        print("Epoch " + str(epochs))
        # Perceptron algorithm
        for i in range(len(x)):
            wx = 0
            for j in range(len(w)):
                wx += x[i][j] * w[j]
            pw = activation_function(wx)
            error = d[i] - pw
            if error != 0:
                for j in range(len(w)):
                    w[j] = w[j] + eta.get() * error * x[i][j]
                done = False
                errors += 1
        error_array.append(errors)
        print("Errors: " + str(errors))
        if not done:
            hyperplane(w, BLUE, "", ':', 0.3)
        else:
            hyperplane(w, BLUE, "Perceptron")
    final_w = w.copy()
    print(final_w)
    error_graph(epochs, error_array, BLUE)


def adaline():
    global done
    global final_w
    error_array = []
    done = False
    epochs = 0
    w = init_w.copy()
    print("Copy of nitial weigths...\n" + str(w))
    print("Desired set..." + str(d))
    print("Training Set..." + str(x))
    norm = np.linalg.norm(x, axis=1)
    print("Norms of Training Set...\n" + str(norm))
    while not done and (epochs < maxEpochs.get()):
        epochs += 1
        error_squared = 0
        print("Epoch: " + str(epochs))
        # Adaline algorithm
        for i in range(len(x)):
            wx = 0
            for j in range(len(w)):
                wx += x[i][j] * w[j]
            pw = activation_function(wx)
            error = d[i] - pw
            error_squared += error * error
            if type_pw.get() == 1:
                for j in range(len(w)):
                    w[j] = w[j] + eta.get() * error * pw * (
                        1 - pw) * x[i][j] / norm[i]
            if type_pw.get() == 2:
                for j in range(len(w)):
                    w[j] = w[j] + eta.get() * error * (
                        1 - pw * pw) * x[i][j] / norm[i]
        error_array.append(error_squared)
        print("Error squared: " + str(error_squared))
        if (error_squared < minError.get()):
            done = True
        if not done:
            if type_pw.get() == 1:
                hyperplane(w, PINK, "", ':', 0.3)
            if type_pw.get() == 2:
                hyperplane(w, ORANGE, "", ':', 0.3)
        else:
            print("Success")
            if type_pw.get() == 1:
                hyperplane(w, PINK, "Adaline (Sigmoid)")
            if type_pw.get() == 2:
                hyperplane(w, ORANGE, "Adaline (Tanh)")
    final_w = w.copy()
    print(final_w)
    if type_pw.get() == 1:
        error_graph(epochs, error_array, PINK)
    if type_pw.get() == 2:
        error_graph(epochs, error_array, ORANGE)
    #conf_matrix = Toplevel(root)
    #conf_matrix.title("Confusion matrix")
    #Label(conf_matrix, text="Confusion matrix").pack()


def start_over():
    global done
    print("Restarting figure...")
    ax_p[0].clear()
    ax_p[1].clear()
    start_graph()
    fig_p.canvas.draw()
    x.clear()
    print("Restarting Training Set...\n" + str(x))
    d.clear()
    print("Restarting Desired Set...\n" + str(d))
    init_w.clear()
    print("Restarting weights...\n" + str(init_w))
    done = False
    b_weights['state'] = 'normal'
    b_perceptron['state'] = 'disabled'
    b_adaline['state'] = 'disabled'


def start_graph():
    # Graph for Perceptron
    major_ticks = np.arange(-5, 6, 1)
    ax_p[0].set_xticks(major_ticks)
    ax_p[0].set_yticks(major_ticks)
    ax_p[0].grid(which='major', linestyle=':', alpha=0.5)
    ax_p[0].axhline(color='white', lw=1)
    ax_p[0].axvline(color='white', lw=1)
    ax_p[0].set_title("Perceptron hyperplane")
    ax_p[0].set_xlim(-5, 5)
    ax_p[0].set_ylim(-5, 5)
    # Graph for errors vs epochs
    ax_p[1].set_xlabel("Epochs")
    ax_p[1].set_ylabel("Errors")


def onclick(event):
    #['#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2', '#fdb462',
    #'#b3de69', '#bc82bd', '#ccebc4', '#ffed6f'])
    global done
    global final_w
    goodPoint = True
    if event.xdata is None or event.ydata is None:
        goodPoint = False
    if event.button == 1 and goodPoint and not done:
        ax_p[0].plot(event.xdata, event.ydata, color=RED, marker='o')
        x.append([-1, event.xdata, event.ydata])
        d.append(0)
    if event.button == 3 and goodPoint and not done:
        ax_p[0].plot(event.xdata, event.ydata, color=GREEN, marker='o')
        x.append([-1, event.xdata, event.ydata])
        d.append(1)
    if done:
        y = (-final_w[1] * event.xdata + final_w[0]) / final_w[2]
        if d[0] == 0:
            colorX = GREEN if y > event.ydata else RED
        else:
            colorX = RED if y > event.ydata else GREEN
        ax_p[0].plot(event.xdata, event.ydata, color=colorX, marker='X')
    fig_p.canvas.draw()


def enable_button():
    state = str(b_weights['state'])
    if type_pw.get() == 0 and state == 'disabled':
        b_perceptron['state'] = 'normal'
        b_adaline['state'] = 'disabled'
    if (type_pw.get() == 1 or type_pw.get() == 2) and state == 'disabled':
        b_perceptron['state'] = 'disabled'
        b_adaline['state'] = 'normal'


# GUI - Initial data
lf_ws = LabelFrame(f_inputData, text="Initial data", padx=5, pady=5)
b_weights = Button(master=lf_ws, text="Random weights", command=init_Weights)

# GUI - Hyperparameters
lf_hyperP = LabelFrame(f_inputData, text="Hyperparameters", padx=5, pady=5)
l_maxEpochs = Label(master=lf_hyperP, text="Max Epochs: ")
e_maxEpochs = Entry(master=lf_hyperP, textvariable=maxEpochs, width=15)
l_minError = Label(master=lf_hyperP, text="Min error: ")
e_minError = Entry(master=lf_hyperP, textvariable=minError, width=15)
s_learningRate = Scale(master=lf_hyperP,
                       label="Learning Rate: ",
                       variable=eta,
                       resolution=0.1,
                       from_=0.1,
                       to=0.9,
                       orient=HORIZONTAL)

# GUI - Activation Function
lf_actF = LabelFrame(f_inputData, text="Activation Function", padx=5, pady=5)
rb_step = Radiobutton(master=lf_actF,
                      text="Perceptron (Step): ",
                      variable=type_pw,
                      value=0,
                      command=enable_button)
rb_sigm = Radiobutton(master=lf_actF,
                      text="Adaline (Sigmoid): ",
                      variable=type_pw,
                      value=1,
                      command=enable_button)
rb_tanh = Radiobutton(master=lf_actF,
                      text="Adaline (Tanh):",
                      variable=type_pw,
                      value=2,
                      command=enable_button)

# GUI - Perceptron
lf_perceptron = LabelFrame(f_inputData, text="Start", padx=5, pady=5)
b_perceptron = Button(master=lf_perceptron,
                      text="Perceptron",
                      command=perceptron,
                      state='disabled')
b_adaline = Button(master=lf_perceptron,
                   text="Adaline",
                   command=adaline,
                   state='disabled')
b_restart = Button(master=lf_perceptron, text="Restart", command=start_over)

# GUI - Canvas
plt.style.use('dark_background')
fig_p, ax_p = plt.subplots(2,
                           figsize=(8, 12),
                           gridspec_kw={'height_ratios': [4, 1]})
start_graph()
cid = fig_p.canvas.mpl_connect('button_press_event', onclick)
canvas_p = FigureCanvasTkAgg(fig_p, master=f_graphs)
canvas_p.get_tk_widget().pack(padx=5, pady=5)

# Packing - Initial data
lf_ws.pack(fill=BOTH, padx=5, pady=5)
b_weights.pack(side=BOTTOM, fill=X, padx=5, pady=5)

# Packing - Hyperparameters
lf_hyperP.pack(fill=BOTH, padx=5, pady=5)
s_learningRate.pack(fill=X, padx=5, pady=5)
l_maxEpochs.pack(anchor=SW)
e_maxEpochs.pack(anchor=NW, padx=5, pady=5)
l_minError.pack(anchor=SW)
e_minError.pack(anchor=NW, padx=5, pady=5)

# Packing - Activation Function
lf_actF.pack(fill=BOTH, padx=5, pady=5)
rb_step.pack(anchor=W, padx=5, pady=5)
rb_sigm.pack(anchor=W, padx=5, pady=5)
rb_tanh.pack(anchor=W, padx=5, pady=5)

# Packing - Perceptron
lf_perceptron.pack(fill=BOTH, padx=5, pady=5)
b_perceptron.pack(fill=BOTH, padx=5, pady=5)
b_adaline.pack(fill=BOTH, padx=5, pady=5)
b_restart.pack(side=BOTTOM, fill=BOTH, padx=5, pady=5)

root.mainloop()
