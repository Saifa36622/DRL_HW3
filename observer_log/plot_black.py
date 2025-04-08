# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # === Replace with your file path ===
# file_path = "q_value/Stabilize/Final/Q_learning/Normal_q/Q_learning_9900_5_12.0_10_20.json"

# def extract_q_values(file_path):
#     with open(file_path, "r") as f:
#         data = json.load(f)

#     q_values = data["q_values"]
#     q_table = {}
#     for key, values in q_values.items():
#         cart_pos, pole_pos, _, _ = eval(key)
#         best_q_value = max(values)
#         if (cart_pos, pole_pos) not in q_table:
#             q_table[(cart_pos, pole_pos)] = best_q_value
#         else:
#             q_table[(cart_pos, pole_pos)] = max(q_table[(cart_pos, pole_pos)], best_q_value)

#     cart_positions = sorted(set(k[0] for k in q_table.keys()))
#     pole_positions = sorted(set(k[1] for k in q_table.keys()))

#     X, Y = np.meshgrid(cart_positions, pole_positions, indexing='ij')
#     Z = np.array([[q_table.get((x, y), np.nan) for y in pole_positions] for x in cart_positions])

#     return X, Y, Z

# def plot_single_q_surface(file_path):
#     X, Y, Z = extract_q_values(file_path)

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.set_xlabel("Cart Position")
#     ax.set_ylabel("Pole Position")
#     ax.set_zlabel("Best Q-Value")
#     ax.set_title("Q-Value Surface")
#     plt.tight_layout()
#     plt.show()

# # === Call the function ===
# plot_single_q_surface(file_path)
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_q_values(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    q_values = data["q_values"]
    q_table = {}
    for key, values in q_values.items():
        cart_pos, pole_pos, _, _ = eval(key)
        best_q_value = max(values)
        if (cart_pos, pole_pos) not in q_table:
            q_table[(cart_pos, pole_pos)] = best_q_value
        else:
            q_table[(cart_pos, pole_pos)] = max(q_table[(cart_pos, pole_pos)], best_q_value)

    cart_positions = sorted(set(k[0] for k in q_table.keys()))
    pole_positions = sorted(set(k[1] for k in q_table.keys()))

    X, Y = np.meshgrid(cart_positions, pole_positions, indexing='ij')
    Z = np.array([[q_table.get((x, y), np.nan) for y in pole_positions] for x in cart_positions])

    return X, Y, Z

def plot_multiple_q_surfaces(file_paths, titles):
    fig = plt.figure(figsize=(18, 6))
    
    for i, (file_path, title) in enumerate(zip(file_paths, titles), start=1):
        X, Y, Z = extract_q_values(file_path)
        ax = fig.add_subplot(1, 3, i, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.view_init(elev=10, azim=0)

        ax.set_xlabel("Cart Position")
        ax.set_ylabel("Pole Position")
        # ax.set_zlabel("Best Q-Value")
        ax.set_title(title)
        # ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

name_compare_1 = "SARSA_discount_0.1"
name_compare_2 = "SARSA_discount_0.99"
# === File paths and titles ===
file_paths = [
    "q_value/Stabilize/Final/SARSA/SARSA_Normal/SARSA_10000_5_12.0_10_20.json",
    # f"q_value/Stabilize/Final/SARSA/{name_compare_1}/SARSA_10000_11_12.0_20_20.json",
    # f"q_value/Stabilize/Final/SARSA/{name_compare_2}/SARSA_10000_3_12.0_5_11.json"
    f"q_value/Stabilize/Final/SARSA/{name_compare_1}/SARSA_10000_5_12.0_10_20.json",
    f"q_value/Stabilize/Final/SARSA/{name_compare_2}/SARSA_10000_5_12.0_10_20.json"
]

titles = [
    "Normal SARSA",
    name_compare_1,
    name_compare_2
]

# === Plot all ===
plot_multiple_q_surfaces(file_paths, titles)
