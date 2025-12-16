import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings('ignore')


class PSOPIDOptimizer:
   """Класс для оптимизации PID-параметров методом роя частиц"""

   def __init__(self, n_particles=30, max_iter=50, w=0.7, c1=1.4, c2=1.4):
       self.n_particles = n_particles  # количество частиц
       self.max_iter = max_iter  # максимальное количество итераций
       self.w = w  # коэффициент инерции
       self.c1 = c1  # коэффициент когнитивной компоненты
       self.c2 = c2  # коэффициент социальной компоненты

       # Границы поиска для параметров [kp_angle, ki_angle, kd_angle, kp_pos, ki_pos, kd_pos]
       self.bounds = np.array([
           [10, 200],  # kp_angle
           [0.1, 10],  # ki_angle
           [5, 50],  # kd_angle
           [0.1, 20],  # kp_pos
           [0.01, 2],  # ki_pos
           [0.1, 15]  # kd_pos
       ])

       # Лучшие найденные параметры
       self.best_params = None
       self.best_fitness = float('inf')

       # История сходимости
       self.convergence_history = []

   def evaluate_fitness(self, params, pendulum_params, initial_conditions):
       """Оценка качества PID-параметров"""
       kp_angle, ki_angle, kd_angle, kp_pos, ki_pos, kd_pos = params

       try:
           # Создаем маятник с тестируемыми параметрами
           pendulum = InvertedPendulum(**pendulum_params)
           pendulum.kp_angle = kp_angle
           pendulum.ki_angle = ki_angle
           pendulum.kd_angle = kd_angle
           pendulum.kp_pos = kp_pos
           pendulum.ki_pos = ki_pos
           pendulum.kd_pos = kd_pos

           # Запускаем симуляцию
           solution = pendulum.simulate(
               initial_state=initial_conditions,
               t_start=0.0,
               t_end=10.0,  # Укороченное время для ускорения оптимизации
               dt=0.01
           )

           if not solution.success:
               return float('inf')

           # Извлекаем данные
           time_array = np.array(pendulum.history['time'])
           theta_array = np.abs(np.array(pendulum.history['theta']))  # Абсолютное значение угла
           x_array = np.abs(np.array(pendulum.history['x']))  # Абсолютное значение позиции
           force_array = np.array(pendulum.history['force'])

           # Критерии качества:
           # 1. Интеграл от абсолютной ошибки угла (IAE)
           iae_angle = np.trapz(theta_array, time_array)

           # 2. Интеграл от абсолютной ошибки позиции (IAE)
           iae_position = np.trapz(x_array, time_array)

           # 3. Максимальное отклонение угла
           max_angle_error = np.max(theta_array)

           # 4. Максимальное отклонение позиции
           max_position_error = np.max(x_array)

           # 5. Энергия управления (интеграл от квадрата силы)
           control_energy = np.trapz(force_array ** 2, time_array)

           # 6. Время установления (время, когда угол становится меньше 5 градусов)
           settling_time_mask = theta_array < 5.0
           if np.any(settling_time_mask):
               settling_time = time_array[np.argmax(settling_time_mask)]
           else:
               settling_time = time_array[-1]  # Если не стабилизировался

           # 7. Штраф за превышение угла 45 градусов
           overshoot_penalty = 1000 if max_angle_error > 45 else 0

           # Комплексная функция приспособленности
           fitness = (
                   0.4 * iae_angle +  # Основной критерий - стабильность угла
                   0.2 * iae_position +  # Стабильность позиции
                   0.1 * max_angle_error +  # Максимальная ошибка угла
                   0.1 * max_position_error +  # Максимальная ошибка позиции
                   0.05 * control_energy +  # Энергия управления
                   0.1 * settling_time +  # Время установления
                   overshoot_penalty  # Штраф за перерегулирование
           )

           # Дополнительный большой штраф если маятник падает (угол > 90 градусов)
           if max_angle_error > 90:
               fitness += 5000

           return fitness

       except Exception as e:
           # В случае ошибки возвращаем плохое значение
           return float('inf')

   def optimize(self, pendulum_params, initial_conditions, verbose=True):
       """Запуск оптимизации методом роя частиц"""

       # Инициализация частиц
       n_dim = 6  # 6 параметров для оптимизации
       particles = np.random.uniform(
           low=self.bounds[:, 0],
           high=self.bounds[:, 1],
           size=(self.n_particles, n_dim)
       )

       velocities = np.random.uniform(-1, 1, (self.n_particles, n_dim))

       # Лучшие позиции частиц
       personal_best_positions = particles.copy()
       personal_best_scores = np.full(self.n_particles, float('inf'))

       # Глобальная лучшая позиция
       global_best_position = None
       global_best_score = float('inf')

       if verbose:
           print("Запуск PSO оптимизации PID-параметров...")
           print(f"Размер роя: {self.n_particles}, Макс. итераций: {self.max_iter}")
           print("=" * 60)

       # Основной цикл оптимизации
       for iteration in range(self.max_iter):
           if verbose:
               print(f"Итерация {iteration + 1}/{self.max_iter}", end="")

           # Оценка приспособленности для всех частиц
           for i in range(self.n_particles):
               fitness = self.evaluate_fitness(particles[i], pendulum_params, initial_conditions)

               # Обновление личных лучших позиций
               if fitness < personal_best_scores[i]:
                   personal_best_scores[i] = fitness
                   personal_best_positions[i] = particles[i].copy()

               # Обновление глобальной лучшей позиции
               if fitness < global_best_score:
                   global_best_score = fitness
                   global_best_position = particles[i].copy()
                   if verbose:
                       print(f" -> Новый лучший результат: {global_best_score:.4f}")

           # Обновление скоростей и позиций
           r1, r2 = np.random.random(2)
           for i in range(self.n_particles):
               # Когнитивная и социальная компоненты
               cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
               social = self.c2 * r2 * (global_best_position - particles[i])

               # Обновление скорости
               velocities[i] = (self.w * velocities[i] + cognitive + social)

               # Обновление позиции
               particles[i] += velocities[i]

               # Ограничение позиций границами поиска
               particles[i] = np.clip(particles[i], self.bounds[:, 0], self.bounds[:, 1])

           # Сохранение истории сходимости
           self.convergence_history.append(global_best_score)

           if verbose and (iteration + 1) % 5 == 0:
               params = global_best_position
               print(f"Итерация {iteration + 1}: Лучшая приспособленность = {global_best_score:.4f}")
               print(f" PID_angle: Kp={params[0]:.2f}, Ki={params[1]:.3f}, Kd={params[2]:.2f}")
               print(f" PID_pos:   Kp={params[3]:.2f}, Ki={params[4]:.3f}, Kd={params[5]:.2f}")
               print("-" * 50)

       # Сохранение лучших параметров
       self.best_params = global_best_position
       self.best_fitness = global_best_score

       if verbose:
           print("\nОптимизация завершена!")
           print("=" * 60)
           print("НАЙДЕННЫЕ ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
           print("=" * 60)
           print(
               f"PID для угла:  Kp={self.best_params[0]:.4f}, Ki={self.best_params[1]:.4f}, Kd={self.best_params[2]:.4f}")
           print(
               f"PID для позиции: Kp={self.best_params[3]:.4f}, Ki={self.best_params[4]:.4f}, Kd={self.best_params[5]:.4f}")
           print(f"Лучшая приспособленность: {self.best_fitness:.4f}")
           print("=" * 60)

       return self.best_params, self.best_fitness

   def plot_convergence(self):
       """Визуализация сходимости алгоритма"""
       plt.figure(figsize=(12, 8))

       plt.subplot(2, 1, 1)
       plt.plot(self.convergence_history, 'b-', linewidth=2)
       plt.title('Сходимость метода роя частиц', fontsize=16, fontweight='bold')
       plt.xlabel('Итерация')
       plt.ylabel('Лучшая приспособленность')
       plt.grid(True, alpha=0.3)

       plt.subplot(2, 1, 2)
       plt.semilogy(self.convergence_history, 'r-', linewidth=2)
       plt.title('Сходимость (логарифмическая шкала)', fontsize=16, fontweight='bold')
       plt.xlabel('Итерация')
       plt.ylabel('Лучшая приспособленность (log)')
       plt.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()


class InvertedPendulum:
   """Класс для симуляции перевернутого маятника с PID-регулятором"""

   def __init__(self, m=1.0, M=5.0, L=1.0, g=9.81, b=0.1, c=0.01):
       # Физические параметры маятника
       self.m = m  # масса груза (кг)
       self.M = M  # масса тележки (кг)
       self.L = L  # длина стержня (м)
       self.g = g  # ускорение свободного падения (м/с²)
       self.b = b  # коэффициент трения тележки
       self.c = c  # коэффициент трения маятника

       # PID параметры для угла маятника
       self.kp_angle = 80.0
       self.ki_angle = 1.0
       self.kd_angle = 15.0

       # PID параметры для положения тележки
       self.kp_pos = 8.0
       self.ki_pos = 0.05
       self.kd_pos = 2.5

       # Ограничение силы (Н)
       self.force_limit = 20.0

       # Целевые значения
       self.target_angle = 0.0
       self.target_position = 0.0

       # Накопленные ошибки
       self.integral_angle = 0.0
       self.integral_position = 0.0
       self.prev_error_angle = 0.0
       self.prev_error_position = 0.0

       # История
       self.history = {
           'time': [], 'x': [], 'x_dot': [], 'theta': [],
           'theta_dot': [], 'force': [], 'error_angle': [], 'error_position': []
       }

   def pid_controller(self, state, dt):
       """PID-регулятор"""
       x, x_dot, theta, theta_dot = state

       # Ошибки
       error_angle = theta - self.target_angle
       error_position = self.target_position - x

       # Интегралы
       self.integral_angle += error_angle * dt
       self.integral_position += error_position * dt

       # Anti-windup
       max_integral_angle = 2.0
       max_integral_position = 2.0

       self.integral_angle = np.clip(self.integral_angle, -max_integral_angle, max_integral_angle)
       self.integral_position = np.clip(self.integral_position, -max_integral_position, max_integral_position)

       # Производные
       if dt > 0:
           derivative_angle = (error_angle - self.prev_error_angle) / dt
           derivative_position = (error_position - self.prev_error_position) / dt
       else:
           derivative_angle = 0.0
           derivative_position = 0.0

       # PID для угла
       force_angle = (self.kp_angle * error_angle +
                      self.ki_angle * self.integral_angle +
                      self.kd_angle * derivative_angle)

       # PID для позиции
       force_position = (self.kp_pos * error_position +
                         self.ki_pos * self.integral_position +
                         self.kd_pos * derivative_position)

       # Комбинированная сила
       force = force_angle + force_position
       force = np.clip(force, -self.force_limit, self.force_limit)

       # Сохранение ошибок
       self.prev_error_angle = error_angle
       self.prev_error_position = error_position

       return force, error_angle, error_position

   def dynamics(self, t, state, dt):
       """Уравнения движения"""
       x, x_dot, theta, theta_dot = state

       force, error_angle, error_position = self.pid_controller(state, dt)

       sin_theta = np.sin(theta)
       cos_theta = np.cos(theta)

       denominator = self.M + self.m * sin_theta ** 2

       # Ускорение тележки
       x_ddot = (force - self.b * x_dot +
                 self.m * self.L * theta_dot ** 2 * sin_theta +
                 self.m * self.g * sin_theta * cos_theta -
                 self.c * theta_dot * cos_theta / self.L) / denominator

       # Угловое ускорение
       theta_ddot = (-(force - self.b * x_dot) * cos_theta -
                     self.m * self.L * theta_dot ** 2 * sin_theta * cos_theta -
                     (self.M + self.m) * self.g * sin_theta +
                     self.c * theta_dot * (self.M + self.m) / (self.m * self.L)) / (self.L * denominator)

       # Сохранение истории
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
       # Очистка
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


def run_optimization_and_simulation():
   """Основная функция для запуска оптимизации и симуляции"""

   # Параметры для оптимизации
   pendulum_params = {
       'm': 1.0,
       'M': 5.0,
       'L': 1.0,
       'g': 9.81,
       'b': 0.1,
       'c': 0.01
   }

   initial_conditions = [0.0, 0.0, np.radians(25.0), 0.0]

   # Создание и запуск оптимизатора
   optimizer = PSOPIDOptimizer(n_particles=20, max_iter=30)

   print("Начинаем оптимизацию PID-параметров...")
   best_params, best_fitness = optimizer.optimize(pendulum_params, initial_conditions)

   # Визуализация сходимости
   optimizer.plot_convergence()

   # Тестирование найденных параметров
   print("\nТЕСТИРОВАНИЕ ОПТИМАЛЬНЫХ ПАРАМЕТРОВ:")
   print("=" * 50)

   # Создаем маятник с оптимальными параметрами
   test_pendulum = InvertedPendulum(**pendulum_params)
   test_pendulum.kp_angle = best_params[0]
   test_pendulum.ki_angle = best_params[1]
   test_pendulum.kd_angle = best_params[2]
   test_pendulum.kp_pos = best_params[3]
   test_pendulum.ki_pos = best_params[4]
   test_pendulum.kd_pos = best_params[5]

   # Запускаем длительную симуляцию для проверки
   test_solution = test_pendulum.simulate(
       initial_state=initial_conditions,
       t_start=0.0,
       t_end=20.0,
       dt=0.01
   )

   # Анализ результатов
   time_array = np.array(test_pendulum.history['time'])
   theta_array = np.array(test_pendulum.history['theta'])
   x_array = np.array(test_pendulum.history['x'])

   # Вычисляем метрики качества
   settling_time_angle = None
   settling_time_position = None

   for i, (t, theta, x) in enumerate(zip(time_array, theta_array, x_array)):
       if settling_time_angle is None and abs(theta) < 2.0:  # Угол стабилизировался в пределах 2 градусов
           settling_time_angle = t
       if settling_time_position is None and abs(x) < 0.1:  # Позиция стабилизировалась в пределах 0.1 м
           settling_time_position = t
       if settling_time_angle is not None and settling_time_position is not None:
           break

   max_angle = np.max(np.abs(theta_array))
   max_position = np.max(np.abs(x_array))

   print(
       f"⏱ Время установления угла: {settling_time_angle:.2f} с" if settling_time_angle else "⏱ Угол не стабилизировался")
   print(
       f"⏱ Время установления позиции: {settling_time_position:.2f} с" if settling_time_position else "⏱ Позиция не стабилизировалась")
   print(f"📐 Максимальное отклонение угла: {max_angle:.2f}°")
   print(f"📍 Максимальное отклонение позиции: {max_position:.3f} м")
   print(f"🎯 Конечная ошибка угла: {abs(theta_array[-1]):.2f}°")
   print(f"🎯 Конечная ошибка позиции: {abs(x_array[-1]):.3f} м")

   # Визуализация результатов
   plt.figure(figsize=(15, 10))

   plt.subplot(2, 2, 1)
   plt.plot(time_array, theta_array, 'b-', linewidth=2)
   plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
   plt.title('Угол маятника', fontsize=14, fontweight='bold')
   plt.xlabel('Время (с)')
   plt.ylabel('Угол (°)')
   plt.grid(True, alpha=0.3)

   plt.subplot(2, 2, 2)
   plt.plot(time_array, x_array, 'g-', linewidth=2)
   plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
   plt.title('Позиция тележки', fontsize=14, fontweight='bold')
   plt.xlabel('Время (с)')
   plt.ylabel('Позиция (м)')
   plt.grid(True, alpha=0.3)

   plt.subplot(2, 2, 3)
   force_array = np.array(test_pendulum.history['force'])
   plt.plot(time_array, force_array, 'r-', linewidth=2)
   plt.title('Управляющая сила', fontsize=14, fontweight='bold')
   plt.xlabel('Время (с)')
   plt.ylabel('Сила (Н)')
   plt.grid(True, alpha=0.3)

   plt.subplot(2, 2, 4)
   # Фазовая плоскость угла
   theta_dot_array = np.array(test_pendulum.history['theta_dot'])
   plt.plot(theta_array, theta_dot_array, 'purple', linewidth=2)
   plt.plot(theta_array[0], theta_dot_array[0], 'go', markersize=10, label='Начало')
   plt.plot(theta_array[-1], theta_dot_array[-1], 'ro', markersize=10, label='Конец')
   plt.title('Фазовая плоскость угла', fontsize=14, fontweight='bold')
   plt.xlabel('Угол (°)')
   plt.ylabel('Угловая скорость (°/с)')
   plt.legend()
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   # Запуск анимации с оптимальными параметрами
   run_animation_with_params(best_params, pendulum_params, initial_conditions)


def run_animation_with_params(pid_params, pendulum_params, initial_conditions):
   """Запуск анимации с заданными параметрами"""

   # Распаковываем параметры
   kp_angle, ki_angle, kd_angle, kp_pos, ki_pos, kd_pos = pid_params

   print("\n🎬 ЗАПУСК АНИМАЦИИ С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ...")
   print("=" * 60)

   # Параметры анимации
   t_start = 0.0
   t_end = 20.0
   dt = 0.01
   ANIMATION_SPEED = 25

   # Создаем маятник с оптимальными параметрами
   pendulum = InvertedPendulum(**pendulum_params)
   pendulum.kp_angle = kp_angle
   pendulum.ki_angle = ki_angle
   pendulum.kd_angle = kd_angle
   pendulum.kp_pos = kp_pos
   pendulum.ki_pos = ki_pos
   pendulum.kd_pos = kd_pos

   # Запуск симуляции
   pendulum.simulate(initial_state=initial_conditions, t_start=t_start, t_end=t_end, dt=dt)

   print("\n" + "=" * 80)
   print("ДЕТАЛЬНАЯ ДИАГНОСТИКА (каждые 0.5 сек):")
   print("=" * 80)
   print(f"{'Время':>6} | {'Позиция':>10} | {'Скорость':>10} | {'Угол':>8} | {'Угл.скор.':>10} | {'Сила':>8}")
   print("-" * 80)

   time_array = np.array(pendulum.history['time'])
   x_array = np.array(pendulum.history['x'])
   x_dot_array = np.array(pendulum.history['x_dot'])
   theta_array = np.array(pendulum.history['theta'])
   theta_dot_array = np.array(pendulum.history['theta_dot'])
   force_array = np.array(pendulum.history['force'])

   # Вывод каждые 0.5 секунды
   interval = 0.5
   for t_check in np.arange(0, t_end + interval, interval):
       # Найти ближайший индекс
       idx = np.argmin(np.abs(time_array - t_check))

       t = time_array[idx]
       x = x_array[idx]
       x_dot = x_dot_array[idx]
       theta = theta_array[idx]
       theta_dot = theta_dot_array[idx]
       force = force_array[idx]

       print(f"{t:6.2f} | {x:+10.4f} | {x_dot:+10.4f} | {theta:+8.2f} | {theta_dot:+10.2f} | {force:+8.2f}")

   print("=" * 80)

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

   # Создание фигуры с настройками отступов
   fig = plt.figure(figsize=(20, 12))
   plt.subplots_adjust(left=0.122, bottom=0.111, right=0.927, top=0.887)

   ax = fig.add_subplot(111)

   # Настройка графика
   ax.set_xlim(x_min, x_max)
   ax.set_ylim(y_min, y_max)
   ax.grid(True, alpha=0.3, linewidth=0.5)
   ax.axhline(y=0, color='k', linewidth=1.5)
   ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
   ax.set_xlabel('Положение (м)', fontsize=18, fontweight='bold')
   ax.set_ylabel('Высота (м)', fontsize=18, fontweight='bold')
   ax.set_title('Анимация маятника с оптимальными PID-параметрами', fontsize=22, fontweight='bold', pad=20)
   ax.tick_params(labelsize=16)

   # Накопленная траектория
   trajectory_line, = ax.plot([], [], 'orange', linewidth=2.5, alpha=0.4, zorder=1)
   cart_trajectory, = ax.plot([], [], 'b-', linewidth=2.5, alpha=0.3, zorder=1)

   # Маркеры траектории
   start_marker, = ax.plot([], [], 'ro', markersize=14, zorder=2,
                           markeredgecolor='darkred', markeredgewidth=3)

   # Текущая анимация
   trail_line, = ax.plot([], [], 'cyan', linewidth=3, alpha=0.8, zorder=3)

   # Тележка
   cart_patch = Rectangle((0, -cart_height / 2), cart_width, cart_height,
                          fill=True, color='blue', alpha=0.9,
                          edgecolor='black', linewidth=3, zorder=4)
   ax.add_patch(cart_patch)

   # Маятник
   pendulum_line, = ax.plot([], [], 'r-', linewidth=5, solid_capstyle='round', zorder=5)
   pendulum_bob, = ax.plot([], [], 'ro', markersize=20, markeredgecolor='darkred',
                           markeredgewidth=3, zorder=6)

   # Текущая позиция
   current_marker, = ax.plot([], [], 'go', markersize=16, zorder=7,
                             markeredgecolor='darkgreen', markeredgewidth=3)

   # Информационный блок
   info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow',
                                 edgecolor='orange', alpha=0.95, linewidth=2.5),
                       family='monospace', fontweight='bold', zorder=8)

   # История следа
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

       # Обновление накопленной траектории
       if frame > 0:
           trajectory_line.set_data(pendulum_x[:frame + 1], pendulum_y[:frame + 1])
           cart_trajectory.set_data(x_positions[:frame + 1], np.zeros(frame + 1))

       # Маркер старта
       start_marker.set_data([pendulum_x[0]], [pendulum_y[0]])

       # Обновление текущей анимации
       # Тележка
       cart_patch.set_x(x_positions[frame] - cart_width / 2)

       # Маятник
       pendulum_line.set_data([x_positions[frame], pendulum_x[frame]],
                              [0, pendulum_y[frame]])
       pendulum_bob.set_data([pendulum_x[frame]], [pendulum_y[frame]])

       # Короткий след за маятником
       trail_x.append(pendulum_x[frame])
       trail_y.append(pendulum_y[frame])
       if len(trail_x) > max_trail:
           trail_x.pop(0)
           trail_y.pop(0)
       trail_line.set_data(trail_x, trail_y)

       # Текущая позиция
       current_marker.set_data([pendulum_x[frame]], [pendulum_y[frame]])

       # Информационный текст
       progress = (frame / len(time_array)) * 100
       info = f'Время: {time_array[frame]:.2f}с [{progress:.0f}%]\n'
       info += f'Позиция: {x_positions[frame]:+.3f} м\n'
       info += f'Угол: {np.degrees(theta_angles[frame]):+.2f}°\n'
       info += f'Ошибка: {abs(x_positions[frame]):.3f} м'
       info_text.set_text(info)

       return cart_patch, pendulum_line, pendulum_bob, trail_line, info_text, \
           trajectory_line, cart_trajectory, start_marker, current_marker

   # Создание анимации
   frames = len(time_array) // ANIMATION_SPEED
   anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=20, blit=True, repeat=False)

   print("✓ Анимация готова! Закройте окно для завершения.")
   plt.show()

   # Финальная статистика
   final_pos_error = abs(pendulum.history['x'][-1])
   final_angle_error = abs(pendulum.history['theta'][-1])

   print("\n" + "=" * 60)
   print("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
   print("=" * 60)
   print(f"Конечная ошибка позиции: {final_pos_error:.3f} м")
   print(f"Конечная ошибка угла: {final_angle_error:.1f}°")

   if final_angle_error < 5 and final_pos_error < 0.5:
       print("✓ ОТЛИЧНО! Маятник успешно стабилизирован!")
   elif final_angle_error < 10 and final_pos_error < 1.0:
       print("~ ХОРОШО. Маятник стабилизирован с небольшими отклонениями.")
   else:
       print("⚠ ПЛОХО. Маятник не стабилизирован должным образом.")
   print("=" * 60)


# Основная программа
if __name__ == "__main__":
   run_optimization_and_simulation()