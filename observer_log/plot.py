# import matplotlib.pyplot as plt
# import json

# # Load your data (assuming it's stored as JSON in a separate file or a long string)
# # For this example, let's assume you've saved the JSON list to a file called `data.json`
# with open("episode_0.json", "r") as f:
#     data = json.load(f)

# # Extract the values for plotting
# cart_pos = [step['cart_pos'] for step in data]
# pole_angle = [step['pole_angle'] for step in data]
# cart_vel = [step['cart_vel'] for step in data]
# pole_vel = [step['pole_vel'] for step in data]
# timesteps = list(range(len(data)))

# # Plotting
# plt.figure(figsize=(10, 5))

# plt.subplot(2, 2, 1)
# plt.plot(timesteps, cart_pos)
# plt.title('Cart Position')
# plt.xlabel('Timestep')
# plt.ylabel('Position (m)')

# plt.subplot(2, 2, 2)
# plt.plot(timesteps, pole_angle)
# plt.title('Pole Angle')
# plt.xlabel('Timestep')
# plt.ylabel('Angle (rad)')

# plt.subplot(2, 2, 3)
# plt.plot(timesteps, cart_vel)
# plt.title('Cart Velocity')
# plt.xlabel('Timestep')
# plt.ylabel('Velocity (m/s)')

# plt.subplot(2, 2, 4)
# plt.plot(timesteps, pole_vel)
# plt.title('Pole Angular Velocity')
# plt.xlabel('Timestep')
# plt.ylabel('Angular Velocity (rad/s)')

# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import json

file_paths = {
    "Normal Q": "episode_0_in_Normal_q.json",
    "Epsilon 0.9996": "episode_0_in_q_epsilon_0.9996.json",
    "Epsilon 0.9998": "episode_0_in_q_epsilon_0.9998.json"
}

data_dict = {}

# Load all three files
for label, path in file_paths.items():
    with open(path, "r") as f:
        data = json.load(f)
        data_dict[label] = {
            'timesteps': list(range(len(data))),
            'cart_pos': [step['cart_pos'] for step in data],
            'pole_angle': [step['pole_angle'] for step in data],
            'cart_vel': [step['cart_vel'] for step in data],
            'pole_vel': [step['pole_vel'] for step in data]
        }

import matplotlib.pyplot as plt
import json

# Load and prepare your data here...

plt.figure(figsize=(12, 8))  # 👈 Smaller size fits most screens

def plot_subplot(index, key, title, ylabel):
    plt.subplot(2, 2, index)
    for label, dataset in data_dict.items():
        plt.plot(dataset['timesteps'], dataset[key], label=label)
    plt.title(title, fontsize=12)
    plt.xlabel('Timestep', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.tick_params(labelsize=9)
    plt.legend(fontsize=9, loc='upper right')  # Move to upper right to avoid overlap

plot_subplot(1, 'cart_pos', 'Cart Position', 'Position (m)')
plot_subplot(2, 'pole_angle', 'Pole Angle', 'Angle (rad)')
plot_subplot(3, 'cart_vel', 'Cart Velocity', 'Velocity (m/s)')
plot_subplot(4, 'pole_vel', 'Pole Angular Velocity', 'Angular Velocity (rad/s)')

plt.tight_layout(pad=2.0)  # 👈 Extra space between plots
plt.show()

