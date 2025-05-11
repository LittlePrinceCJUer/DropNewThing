import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_arch(layers, title):
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.axis('off')

    # Rectangle width (percentage of plot width)
    rect_width = 0.9
    x_start = (1 - rect_width) / 2

    # Draw each layer as a rectangle with text
    for text, y in layers:
        rect = patches.Rectangle(
            (x_start, y - 0.03), rect_width, 0.06,
            fill=False, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(0.5, y, text, ha='center', va='center', fontsize=10)

    # Draw arrows between layers
    for i in range(len(layers) - 1):
        y1 = layers[i][1] - 0.03
        y2 = layers[i + 1][1] + 0.03
        ax.annotate(
            "",
            xy=(0.5, y2),
            xytext=(0.5, y1),
            arrowprops=dict(arrowstyle='->')
        )

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# # MNIST_CNN
# mnist_layers = [
#     ("Conv1: ch 1→6, k 3×3, p 1, s 1, ReLU", 0.9),
#     ("MaxPool1: 2×2", 0.8),
#     ("Conv2: ch 6→10, k 3×3, p 0, s 1, ReLU", 0.7),
#     ("MaxPool2: 2×2", 0.6),
#     ("Conv3: ch 10→24, k 3×3, p 0, s 1, ReLU", 0.5),
#     ("MaxPool3: 2×2", 0.4),
#     ("FC1: 96→64, ReLU", 0.3),
#     ("FC2: 64→10, Softmax", 0.2)
# ]
# draw_arch(mnist_layers, "MNIST_CNN")

# # MNIST_CNN_add1
# mnist_layers = [
#     ("Conv1: ch 1→6, k 3×3, p 1, s 1, ReLU", 0.9),
#     ("MaxPool1: 2×2", 0.8),
#     ("Conv2: ch 6→10, k 3×3, p 0, s 1, ReLU", 0.7),
#     ("MaxPool2: 2×2", 0.6),
#     ("Conv3: ch 10→24, k 3×3, p 0, s 1, ReLU", 0.5),
#     ("MaxPool3: 2×2", 0.4),
#     ("Conv4: ch 24→64, k 3×3, p 1, s 1, ReLU", 0.34),
#     ("MaxPool4: 2×2", 0.26),
#     ("FC1: 64→64, ReLU", 0.18),
#     ("FC2: 64→10, Softmax", 0.1)
# ]
# draw_arch(mnist_layers, "MNIST_CNN_add1")

# # MNIST_CNN_del1
# mnist_layers = [
#     ("Conv1: ch 1→6, k 3×3, p 1, s 1, ReLU", 0.9),
#     ("MaxPool1: 2×2", 0.75),
#     ("Conv2: ch 6→10, k 3×3, p 0, s 1, ReLU", 0.6),
#     ("MaxPool2: 2×2", 0.45),
#     # ("Conv3: ch 10→24, k 3×3, p 0, s 1, ReLU", 0.5),
#     # ("MaxPool3: 2×2", 0.4),
#     # ("Conv4: ch 24→64, k 3×3, p 1, s 1, ReLU", 0.34),
#     # ("MaxPool4: 2×2", 0.26),
#     ("FC1: 360→64, ReLU", 0.3),
#     ("FC2: 64→10, Softmax", 0.15)
# ]
# draw_arch(mnist_layers, "MNIST_CNN_del1")

# # CIFAR10_CNN
# cifar_layers = [
#     ("Conv1: ch 3→32, k 3×3, p 1, s 1, ReLU", 0.95),
#     ("MaxPool: 2×2", 0.88),
#     ("Conv2: ch 32→64, k 3×3, p 1, s 1, ReLU ", 0.81),
#     ("MaxPool: 2×2", 0.74),
#     ("Conv3: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.67),
#     ("MaxPool: 2×2", 0.60),
#     ("Conv4: ch 128→256, k 3×3, p 1, s 1, ReLU", 0.53),
#     ("MaxPool: 2×2", 0.46),
#     ("FC1: 1024→128, ReLU", 0.32),
#     ("FC2: 128→10, Softmax", 0.20)
# ]
# draw_arch(cifar_layers, "CIFAR10_CNN")

# CIFAR10_CNN_add1
cifar_layers = [
    ("Conv1: ch 3→32, k 3×3, p 1, s 1, ReLU", 0.92),
    ("MaxPool: 2×2", 0.84),
    ("Conv2: ch 32→64, k 3×3, p 1, s 1, ReLU ", 0.76),
    ("MaxPool: 2×2", 0.68),
    ("Conv3: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.6),
    ("MaxPool: 2×2", 0.52),
    ("Conv4: ch 128→256, k 3×3, p 1, s 1, ReLU", 0.44),
    ("MaxPool: 2×2", 0.36),
    ("Conv5: ch 256→512, k 3×3, p 1, s 1, ReLU", 0.28),
    ("MaxPool: 2×2", 0.2),
    ("FC1: 512→128, ReLU", 0.12),
    ("FC2: 128→10, Softmax", 0.04)
]
draw_arch(cifar_layers, "CIFAR10_CNN_add1")

# CIFAR10_CNN_del1
cifar_layers = [
    ("Conv1: ch 3→32, k 3×3, p 1, s 1, ReLU", 0.92),
    ("MaxPool: 2×2", 0.84),
    ("Conv2: ch 32→64, k 3×3, p 1, s 1, ReLU ", 0.76),
    ("MaxPool: 2×2", 0.68),
    ("Conv3: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.6),
    ("MaxPool: 2×2", 0.52),
    # ("Conv4: ch 128→256, k 3×3, p 1, s 1, ReLU", 0.44),
    # ("MaxPool: 2×2", 0.36),
    # ("Conv5: ch 256→512, k 3×3, p 1, s 1, ReLU", 0.28),
    # ("MaxPool: 2×2", 0.2),
    ("FC1: 2048→128, ReLU", 0.12),
    ("FC2: 128→10, Softmax", 0.04)
]
draw_arch(cifar_layers, "CIFAR10_CNN_del1")

# # ORL_CNN
# orl_layers = [
#     ("Conv1: ch 1→3, k 3×3, p 1, s 1, ReLU", 0.95),
#     ("MaxPool: 2×2", 0.88),
#     ("Conv2: ch →32, k 3×3, p 1, s 1, ReLU", 0.81),
#     ("MaxPool: 2×2", 0.74),
#     ("Conv3: ch 32→64, k 3×3, p 1, s 1, ReLU", 0.67),
#     ("MaxPool: 2×2", 0.60),
#     ("Conv4: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.53),
#     ("MaxPool: 2×2", 0.46),
#     ("FC1: 4480→128, ReLU", 0.32),
#     ("FC2: 128→40, Softmax", 0.20)
# ]

# ORL_CNN_add1
orl_layers = [
    ("Conv1: ch 1→3, k 3×3, p 1, s 1, ReLU", 0.92),
    ("MaxPool: 2×2", 0.84),
    ("Conv2: ch →32, k 3×3, p 1, s 1, ReLU", 0.76),
    ("MaxPool: 2×2", 0.68),
    ("Conv3: ch 32→64, k 3×3, p 1, s 1, ReLU", 0.6),
    ("MaxPool: 2×2", 0.52),
    ("Conv4: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.44),
    ("MaxPool: 2×2", 0.36),
    ("Conv5: ch 128→256, k 3×3, p 1, s 1, ReLU", 0.28),
    ("MaxPool: 2×2", 0.2),
    ("FC1: 1536→128, ReLU", 0.12),
    ("FC2: 128→40, Softmax", 0.04)
]
draw_arch(orl_layers, "ORL_CNN_add1")

# ORL_CNN_del1
orl_layers = [
    ("Conv1: ch 1→3, k 3×3, p 1, s 1, ReLU", 0.92),
    ("MaxPool: 2×2", 0.84),
    ("Conv2: ch →32, k 3×3, p 1, s 1, ReLU", 0.76),
    ("MaxPool: 2×2", 0.68),
    ("Conv3: ch 32→64, k 3×3, p 1, s 1, ReLU", 0.6),
    ("MaxPool: 2×2", 0.52),
    # ("Conv4: ch 64→128, k 3×3, p 1, s 1, ReLU", 0.44),
    # ("MaxPool: 2×2", 0.36),
    # ("Conv5: ch 128→256, k 3×3, p 1, s 1, ReLU", 0.28),
    # ("MaxPool: 2×2", 0.2),
    ("FC1: 9856→128, ReLU", 0.12),
    ("FC2: 128→40, Softmax", 0.04)
]
draw_arch(orl_layers, "ORL_CNN_del1")
