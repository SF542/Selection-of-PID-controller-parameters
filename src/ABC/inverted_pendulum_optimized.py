import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class InvertedPendulum:
    """Класс для симуляции перевернутого маятника с PID-регулятором"""

    def __init__(self, m=1.0, M=5.0, L=1.0, g=9.81, b=0.1, c=0.01):
        self.m = m
        self.M = M
        self.L = L
        self.g = g
        self.b = b
        self.c = c

        self.kp_angle = 80.0
        self.ki_angle = 1.0
        self.kd_angle = 15.0
        self.kp_pos = 8.0
        self.ki_pos = 0.05
        self.kd_pos = 2.5

        self.force_limit = 20.0
        self.target_angle = 0.0
        self.target_position = 0.0

        self.integral_angle = 0.0
        self.integral_position = 0.0
        self.prev_error_angle = 0.0
        self.prev_error_position = 0.0

        self.history = {
            'time': [], 'x': [], 'x_dot': [], 'theta': [],
            'theta_dot': [], 'force': [], 'error_angle': [], 'error_position': []
        }

    def pid_controller(self, state, dt):
        """PID-регулятор"""
        x, x_dot, theta, theta_dot = state

        error_angle = theta - self.target_angle
        error_position = self.target_position - x

        self.integral_angle += error_angle * dt
        self.integral_position += error_position * dt

        max_integral_angle = 2.0
        max_integral_position = 2.0

        self.integral_angle = np.clip(self.integral_angle, -max_integral_angle, max_integral_angle)
        self.integral_position = np.clip(self.integral_position, -max_integral_position, max_integral_position)

        if dt > 0:
            derivative_angle = (error_angle - self.prev_error_angle) / dt
            derivative_position = (error_position - self.prev_error_position) / dt
        else:
            derivative_angle = 0.0
            derivative_position = 0.0

        force_angle = (self.kp_angle * error_angle +
                      self.ki_angle * self.integral_angle +
                      self.kd_angle * derivative_angle)

        force_position = (self.kp_pos * error_position +
                         self.ki_pos * self.integral_position +
                         self.kd_pos * derivative_position)

        force = force_angle + force_position
        force = np.clip(force, -self.force_limit, self.force_limit)

        self.prev_error_angle = error_angle
        self.prev_error_position = error_position

        return force, error_angle, error_position

    def dynamics(self, t, state, dt):
        """Уравнения движения"""
        x, x_dot, theta, theta_dot = state

        force, error_angle, error_position = self.pid_controller(state, dt)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        denominator = self.M + self.m * sin_theta**2

        x_ddot = (force - self.b * x_dot +
                 self.m * self.L * theta_dot**2 * sin_theta +
                 self.m * self.g * sin_theta * cos_theta -
                 self.c * theta_dot * cos_theta / self.L) / denominator

        theta_ddot = (-(force - self.b * x_dot) * cos_theta -
                     self.m * self.L * theta_dot**2 * sin_theta * cos_theta -
                     (self.M + self.m) * self.g * sin_theta +
                     self.c * theta_dot * (self.M + self.m) / (self.m * self.L)) / (self.L * denominator)

        self.history['time'].append(t)
        self.history['x'].append(x)
        self.history['x_dot'].append(x_dot)
        self.history['theta'].append(np.degrees(theta))
        self.history['theta_dot'].append(np.degrees(theta_dot))
        self.history['force'].append(force)
        self.history['error_angle'].append(np.degrees(error_angle))
        self.history['error_position'].append(error_position)

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def simulate(self, initial_state, t_start=0.0, t_end=10.0, dt=0.01):
        """Запуск симуляции"""
        self.history = {key: [] for key in self.history.keys()}
        self.integral_angle = 0.0
        self.integral_position = 0.0
        self.prev_error_angle = 0.0
        self.prev_error_position = 0.0

        t_eval = np.arange(t_start, t_end, dt)

        solution = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, dt),
            t_span=(t_start, t_end),
            y0=initial_state,
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )

        return solution


# Физические параметры
m = 1.0      
M = 5.0      
L = 1.0      
g = 9.81     
b = 0.1      
c = 0.01  

kp_angle = 50.0
ki_angle = 1.0    
kd_angle = 15.0
kp_pos = 3.0
ki_pos = 0.05      
kd_pos = 5.0

# Время симуляции
t_start = 0.0
t_end = 20.0       
dt = 0.01

# Начальные условия
INITIAL_POSITION = 0.0
INITIAL_ANGLE = 25.0  
ANIMATION_SPEED = 25

print("Запуск симуляции...")
pendulum = InvertedPendulum(m=m, M=M, L=L, g=g, b=b, c=c)
pendulum.kp_angle = kp_angle
pendulum.ki_angle = ki_angle
pendulum.kd_angle = kd_angle
pendulum.kp_pos = kp_pos
pendulum.ki_pos = ki_pos
pendulum.kd_pos = kd_pos

initial_state = [
    INITIAL_POSITION,
    0.0,
    np.radians(INITIAL_ANGLE),
    0.0
]

pendulum.simulate(initial_state=initial_state, t_start=t_start, t_end=t_end, dt=dt)

print("\n" + "="*80)
print("ДЕТАЛЬНАЯ ДИАГНОСТИКА (каждые 0.5 сек):")
print("="*80)
print(f"{'Время':>6} | {'Позиция':>10} | {'Скорость':>10} | {'Угол':>8} | {'Угл.скор.':>10} | {'Сила':>8}")
print("-"*80)

time_array = np.array(pendulum.history['time'])
x_array = np.array(pendulum.history['x'])
x_dot_array = np.array(pendulum.history['x_dot'])
theta_array = np.array(pendulum.history['theta'])
theta_dot_array = np.array(pendulum.history['theta_dot'])
force_array = np.array(pendulum.history['force'])

interval = 0.5
for t_check in np.arange(0, t_end + interval, interval):
    idx = np.argmin(np.abs(time_array - t_check))
    
    t = time_array[idx]
    x = x_array[idx]
    x_dot = x_dot_array[idx]
    theta = theta_array[idx]
    theta_dot = theta_dot_array[idx]
    force = force_array[idx]
    
    print(f"{t:6.2f} | {x:+10.4f} | {x_dot:+10.4f} | {theta:+8.2f} | {theta_dot:+10.2f} | {force:+8.2f}")

print("="*80)

# Построение графиков
print("\nПостроение графиков...")

fig_plots, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95, top=0.94, hspace=0.35)

ax1.plot(time_array, theta_array, 'b-', linewidth=2.5, label='Угол (°)')
ax1.axhline(y=2, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Допуск ±2°')
ax1.axhline(y=-2, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.fill_between(time_array, -2, 2, color='green', alpha=0.1)
ax1.set_xlabel('Время (с)', fontsize=13, fontweight='normal')
ax1.set_ylabel('Угол (°)', fontsize=13, fontweight='normal')
ax1.set_title('Угол маятника', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.3, linewidth=0.8)
ax1.legend(fontsize=11, loc='upper right')
ax1.tick_params(labelsize=11)
ax1.set_xlim(0, t_end)

ax2.plot(time_array, force_array, 'darkorange', linewidth=2.5, label='Сила (Н)')
ax2.axhline(y=pendulum.force_limit, color='r', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Ограничение ±{pendulum.force_limit} Н')
ax2.axhline(y=-pendulum.force_limit, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax2.set_xlabel('Время (с)', fontsize=13, fontweight='normal')
ax2.set_ylabel('Сила (Н)', fontsize=13, fontweight='normal')
ax2.set_title('Сила управления', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.3, linewidth=0.8)
ax2.legend(fontsize=11, loc='upper right')
ax2.tick_params(labelsize=11)
ax2.set_xlim(0, t_end)

plt.savefig('control_graphs.png', dpi=300, bbox_inches='tight')
print("Графики сохранены: control_graphs.png")
plt.show()

# Подготовка данных для анимации
x_positions = np.array(pendulum.history['x'])
theta_angles = np.deg2rad(np.array(pendulum.history['theta']))
time_array = np.array(pendulum.history['time'])

pendulum_x = x_positions + pendulum.L * np.sin(theta_angles)
pendulum_y = pendulum.L * np.cos(theta_angles)

x_min = min(x_positions.min(), pendulum_x.min()) - 1
x_max = max(x_positions.max(), pendulum_x.max()) + 1
y_min = -0.5
y_max = pendulum.L + 0.5

cart_width, cart_height = 0.4, 0.2

fig = plt.figure(figsize=(20, 12))
plt.subplots_adjust(left=0.122, bottom=0.111, right=0.927, top=0.887)

ax = fig.add_subplot(111)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.axhline(y=0, color='k', linewidth=1.5)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('Положение (м)', fontsize=18, fontweight='bold')
ax.set_ylabel('Высота (м)', fontsize=18, fontweight='bold')
ax.set_title('Анимация маятника с накопленной траекторией', fontsize=22, fontweight='bold', pad=20)
ax.tick_params(labelsize=16)

trajectory_line, = ax.plot([], [], 'orange', linewidth=2.5, alpha=0.4, zorder=1)
cart_trajectory, = ax.plot([], [], 'b-', linewidth=2.5, alpha=0.3, zorder=1)

start_marker, = ax.plot([], [], 'ro', markersize=14, zorder=2, 
                        markeredgecolor='darkred', markeredgewidth=3)

trail_line, = ax.plot([], [], 'cyan', linewidth=3, alpha=0.8, zorder=3)

cart_patch = Rectangle((0, -cart_height/2), cart_width, cart_height,
                       fill=True, color='blue', alpha=0.9,
                       edgecolor='black', linewidth=3, zorder=4)
ax.add_patch(cart_patch)

pendulum_line, = ax.plot([], [], 'r-', linewidth=5, solid_capstyle='round', zorder=5)
pendulum_bob, = ax.plot([], [], 'ro', markersize=20, markeredgecolor='darkred', 
                       markeredgewidth=3, zorder=6)

current_marker, = ax.plot([], [], 'go', markersize=16, zorder=7, 
                          markeredgecolor='darkgreen', markeredgewidth=3)

info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                             edgecolor='orange', alpha=0.95, linewidth=2.5),
                    family='monospace', fontweight='bold', zorder=8)

trail_x, trail_y = [], []
max_trail = 40

def init():
    cart_patch.set_x(-100)
    pendulum_line.set_data([], [])
    pendulum_bob.set_data([], [])
    trail_line.set_data([], [])
    info_text.set_text('')
    trajectory_line.set_data([], [])
    cart_trajectory.set_data([], [])
    start_marker.set_data([], [])
    current_marker.set_data([], [])
    return cart_patch, pendulum_line, pendulum_bob, trail_line, info_text, \
           trajectory_line, cart_trajectory, start_marker, current_marker

def update(frame):
    frame = frame * ANIMATION_SPEED
    if frame >= len(time_array):
        frame = len(time_array) - 1
    
    if frame > 0:
        trajectory_line.set_data(pendulum_x[:frame+1], pendulum_y[:frame+1])
        cart_trajectory.set_data(x_positions[:frame+1], np.zeros(frame+1))
    
    start_marker.set_data([pendulum_x[0]], [pendulum_y[0]])
    
    cart_patch.set_x(x_positions[frame] - cart_width/2)
    
    pendulum_line.set_data([x_positions[frame], pendulum_x[frame]],
                          [0, pendulum_y[frame]])
    pendulum_bob.set_data([pendulum_x[frame]], [pendulum_y[frame]])
    
    trail_x.append(pendulum_x[frame])
    trail_y.append(pendulum_y[frame])
    if len(trail_x) > max_trail:
        trail_x.pop(0)
        trail_y.pop(0)
    trail_line.set_data(trail_x, trail_y)
    
    current_marker.set_data([pendulum_x[frame]], [pendulum_y[frame]])
    
    progress = (frame / len(time_array)) * 100
    info = f'Время: {time_array[frame]:.2f}с [{progress:.0f}%]\n'
    info += f'Позиция: {x_positions[frame]:+.3f} м\n'
    info += f'Угол: {np.degrees(theta_angles[frame]):+.2f}°\n'
    info += f'Ошибка: {abs(x_positions[frame]):.3f} м'
    info_text.set_text(info)
    
    return cart_patch, pendulum_line, pendulum_bob, trail_line, info_text, \
           trajectory_line, cart_trajectory, start_marker, current_marker

frames = len(time_array) // ANIMATION_SPEED
anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                    interval=20, blit=True, repeat=False)

print("Анимация готова! Закройте окно для завершения.")

plt.show()

# Финальная статистика
final_pos_error = abs(pendulum.history['x'][-1])
final_angle_error = abs(pendulum.history['theta'][-1])

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
print("="*60)
print(f"Конечная ошибка позиции: {final_pos_error:.3f} м")
print(f"Конечная ошибка угла: {final_angle_error:.1f}°")

if final_angle_error < 5 and final_pos_error < 0.5:
    print("ОТЛИЧНО! Маятник успешно стабилизирован!")
elif final_angle_error < 10 and final_pos_error < 1.0:
    print("ХОРОШО. Маятник стабилизирован с небольшими отклонениями.")
else:
    print("ПЛОХО. Маятник не стабилизирован должным образом.")
print("="*60)