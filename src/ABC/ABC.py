"""
Artificial Bee Colony (ABC) Algorithm для автоматической настройки PID-параметров
перевернутого маятника.

Алгоритм имитирует поведение пчелиной колонии:
- Employed Bees (работники): исследуют текущие источники нектара
- Onlooker Bees (наблюдатели): выбирают лучшие источники
- Scout Bees (разведчики): ищут новые области
"""

import numpy as np
import matplotlib.pyplot as plt
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

        self.kp_angle = None
        self.ki_angle = None
        self.kd_angle = None
        self.kp_pos = None
        self.ki_pos = None
        self.kd_pos = None

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

        try:
            solution = solve_ivp(
                fun=lambda t, y: self.dynamics(t, y, dt),
                t_span=(t_start, t_end),
                y0=initial_state,
                t_eval=t_eval,
                method='RK45',
                max_step=dt
            )
            return solution
        except:
            return None


class ArtificialBeeColony:
    """Artificial Bee Colony (ABC) Algorithm для оптимизации PID-параметров"""

    def __init__(self, colony_size=30, max_iterations=50, limit=10, bounds=None):
        """
        Параметры:
        - colony_size: размер колонии
        - max_iterations: максимальное количество итераций
        - limit: лимит попыток улучшения источника
        - bounds: границы параметров [[min, max], ...]
        """
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit
        
        if bounds is None:
            self.bounds = np.array([
                [10.0, 100.0],   # kp_angle
                [0.0, 5.0],      # ki_angle
                [5.0, 30.0],     # kd_angle
                [0.0, 10.0],     # kp_pos
                [0.0, 2.0],      # ki_pos
                [0.0, 10.0]      # kd_pos
            ])
        else:
            self.bounds = np.array(bounds)
        
        self.dimension = len(self.bounds)
        
        self.population = None
        self.fitness = None
        self.trial_counter = None
        
        self.best_solution = None
        self.best_fitness = float('inf')
        
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_solution': []
        }

    def initialize_population(self):
        """Инициализация начальной популяции"""
        self.population = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.colony_size, self.dimension)
        )
        self.fitness = np.full(self.colony_size, float('inf'))
        self.trial_counter = np.zeros(self.colony_size)

    def evaluate_fitness(self, solution, pendulum_params, initial_conditions, t_end=15.0):
        """Оценка качества решения"""
        pendulum = InvertedPendulum(**pendulum_params)
        
        pendulum.kp_angle = solution[0]
        pendulum.ki_angle = solution[1]
        pendulum.kd_angle = solution[2]
        pendulum.kp_pos = solution[3]
        pendulum.ki_pos = solution[4]
        pendulum.kd_pos = solution[5]
        
        result = pendulum.simulate(
            initial_state=initial_conditions,
            t_start=0.0,
            t_end=t_end,
            dt=0.01
        )
        
        if result is None or not result.success:
            return 1e6
        
        final_x = pendulum.history['x'][-1]
        final_theta = pendulum.history['theta'][-1]
        
        if abs(final_x) > 10.0 or abs(final_theta) > 180.0:
            return 1e6
        
        final_position_error = abs(pendulum.history['x'][-1])
        final_angle_error = abs(pendulum.history['theta'][-1])
        
        last_20_percent = int(len(pendulum.history['x']) * 0.8)
        avg_position_error = np.mean(np.abs(pendulum.history['x'][last_20_percent:]))
        avg_angle_error = np.mean(np.abs(pendulum.history['theta'][last_20_percent:]))
        
        settling_time = t_end
        for i, (theta, x) in enumerate(zip(pendulum.history['theta'], pendulum.history['x'])):
            if abs(theta) < 5.0 and abs(x) < 0.5:
                settling_time = pendulum.history['time'][i]
                break
        
        w1 = 100.0
        w2 = 10.0
        w3 = 50.0
        w4 = 5.0
        w5 = 2.0
        
        fitness = (w1 * final_angle_error**2 + 
                  w2 * final_position_error**2 +
                  w3 * avg_angle_error**2 +
                  w4 * avg_position_error**2 +
                  w5 * settling_time)
        
        return fitness

    def employed_bee_phase(self, pendulum_params, initial_conditions, t_end):
        """Фаза работающих пчел"""
        for i in range(self.colony_size):
            phi = np.random.uniform(-1, 1, self.dimension)
            
            partner_idx = i
            while partner_idx == i:
                partner_idx = np.random.randint(0, self.colony_size)
            
            new_solution = self.population[i] + phi * (self.population[i] - self.population[partner_idx])
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            
            new_fitness = self.evaluate_fitness(new_solution, pendulum_params, initial_conditions, t_end)
            
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1

    def calculate_selection_probabilities(self):
        """Вычисление вероятностей выбора источников"""
        fitness_quality = 1.0 / (self.fitness + 1e-10)
        probabilities = fitness_quality / np.sum(fitness_quality)
        return probabilities

    def onlooker_bee_phase(self, pendulum_params, initial_conditions, t_end):
        """Фаза пчел-наблюдателей"""
        probabilities = self.calculate_selection_probabilities()
        
        for _ in range(self.colony_size):
            selected_idx = np.random.choice(self.colony_size, p=probabilities)
            
            phi = np.random.uniform(-1, 1, self.dimension)
            
            partner_idx = selected_idx
            while partner_idx == selected_idx:
                partner_idx = np.random.randint(0, self.colony_size)
            
            new_solution = self.population[selected_idx] + phi * (self.population[selected_idx] - self.population[partner_idx])
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            
            new_fitness = self.evaluate_fitness(new_solution, pendulum_params, initial_conditions, t_end)
            
            if new_fitness < self.fitness[selected_idx]:
                self.population[selected_idx] = new_solution
                self.fitness[selected_idx] = new_fitness
                self.trial_counter[selected_idx] = 0
            else:
                self.trial_counter[selected_idx] += 1

    def scout_bee_phase(self):
        """Фаза пчел-разведчиков"""
        for i in range(self.colony_size):
            if self.trial_counter[i] > self.limit:
                self.population[i] = np.random.uniform(
                    low=self.bounds[:, 0],
                    high=self.bounds[:, 1],
                    size=self.dimension
                )
                self.fitness[i] = float('inf')
                self.trial_counter[i] = 0
                print(f"   Scout bee: заменен источник {i}")

    def update_best_solution(self):
        """Обновление лучшего решения"""
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()

    def optimize(self, pendulum_params, initial_conditions, t_end=15.0, verbose=True):
        """Запуск оптимизации ABC алгоритмом"""
        print("="*80)
        print("ЗАПУСК ARTIFICIAL BEE COLONY ОПТИМИЗАЦИИ")
        print("="*80)
        print(f"Параметры алгоритма:")
        print(f"  Размер колонии: {self.colony_size}")
        print(f"  Максимум итераций: {self.max_iterations}")
        print(f"  Лимит попыток: {self.limit}")
        print(f"  Размерность: {self.dimension}")
        print(f"Границы параметров:")
        param_names = ['kp_angle', 'ki_angle', 'kd_angle', 'kp_pos', 'ki_pos', 'kd_pos']
        for i, name in enumerate(param_names):
            print(f"  {name}: [{self.bounds[i][0]:.1f}, {self.bounds[i][1]:.1f}]")
        print("="*80)
        
        self.initialize_population()
        
        if verbose:
            print("\nИнициализация популяции...")
        for i in range(self.colony_size):
            self.fitness[i] = self.evaluate_fitness(
                self.population[i], 
                pendulum_params, 
                initial_conditions, 
                t_end
            )
        
        self.update_best_solution()
        
        if verbose:
            print(f"Начальная лучшая fitness: {self.best_fitness:.4f}")
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Итерация {iteration + 1}/{self.max_iterations}")
                print(f"{'='*80}")
            
            if verbose:
                print("   Employed Bee Phase...")
            self.employed_bee_phase(pendulum_params, initial_conditions, t_end)
            
            if verbose:
                print("   Onlooker Bee Phase...")
            self.onlooker_bee_phase(pendulum_params, initial_conditions, t_end)
            
            if verbose:
                print("   Scout Bee Phase...")
            self.scout_bee_phase()
            
            self.update_best_solution()
            
            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_fitness'].append(np.mean(self.fitness))
            self.history['best_solution'].append(self.best_solution.copy())
            
            if verbose:
                print(f"\n   Результаты итерации:")
                print(f"    Лучшая fitness: {self.best_fitness:.4f}")
                print(f"    Средняя fitness: {np.mean(self.fitness):.4f}")
                print(f"    Лучшее решение:")
                for i, name in enumerate(param_names):
                    print(f"      {name}: {self.best_solution[i]:.4f}")
            
            if self.best_fitness < 1.0:
                print(f"\nДостигнут отличный результат! Остановка на итерации {iteration + 1}")
                break
        
        print("\n" + "="*80)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print("="*80)
        print(f"Лучшая найденная fitness: {self.best_fitness:.4f}")
        print(f"Оптимальные PID параметры:")
        for i, name in enumerate(param_names):
            print(f"  {name} = {self.best_solution[i]:.4f}")
        print("="*80)
        
        return self.best_solution, self.best_fitness

    def plot_convergence(self):
        """Визуализация процесса сходимости"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        iterations = range(1, len(self.history['best_fitness']) + 1)
        
        ax1.plot(iterations, self.history['best_fitness'], 'b-', linewidth=2, label='Лучшая fitness')
        ax1.plot(iterations, self.history['mean_fitness'], 'r--', linewidth=2, label='Средняя fitness')
        ax1.set_xlabel('Итерация', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax1.set_title('Сходимость ABC алгоритма', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_yscale('log')
        
        param_names = ['kp_angle', 'ki_angle', 'kd_angle', 'kp_pos', 'ki_pos', 'kd_pos']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, color) in enumerate(zip(param_names, colors)):
            values = [sol[i] for sol in self.history['best_solution']]
            ax2.plot(iterations, values, color=color, linewidth=2, label=name, marker='o', markersize=4)
        
        ax2.set_xlabel('Итерация', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Значение параметра', fontsize=12, fontweight='bold')
        ax2.set_title('Эволюция PID параметров', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        return fig


def test_optimized_parameters(best_solution, pendulum_params, initial_conditions, t_end=20.0):
    """Тестирование оптимизированных параметров"""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННЫХ ПАРАМЕТРОВ")
    print("="*80)
    
    pendulum = InvertedPendulum(**pendulum_params)
    pendulum.kp_angle = best_solution[0]
    pendulum.ki_angle = best_solution[1]
    pendulum.kd_angle = best_solution[2]
    pendulum.kp_pos = best_solution[3]
    pendulum.ki_pos = best_solution[4]
    pendulum.kd_pos = best_solution[5]
    
    result = pendulum.simulate(
        initial_state=initial_conditions,
        t_start=0.0,
        t_end=t_end,
        dt=0.01
    )
    
    if result is None or not result.success:
        print("Симуляция не удалась!")
        return None
    
    final_pos_error = abs(pendulum.history['x'][-1])
    final_angle_error = abs(pendulum.history['theta'][-1])
    
    settling_time = t_end
    for i, (theta, x) in enumerate(zip(pendulum.history['theta'], pendulum.history['x'])):
        if abs(theta) < 5.0 and abs(x) < 0.5:
            settling_time = pendulum.history['time'][i]
            break
    
    print(f"\nРезультаты симуляции:")
    print(f"  Конечная ошибка позиции: {final_pos_error:.4f} м")
    print(f"  Конечная ошибка угла: {final_angle_error:.2f}°")
    print(f"  Время стабилизации: {settling_time:.2f} сек")
    
    if final_angle_error < 5 and final_pos_error < 0.5:
        print("\nОТЛИЧНО! Маятник успешно стабилизирован!")
    elif final_angle_error < 10 and final_pos_error < 1.0:
        print("\nХОРОШО. Маятник стабилизирован с небольшими отклонениями.")
    else:
        print("\nПЛОХО. Маятник не стабилизирован должным образом.")
    
    print("="*80)
    
    return pendulum


if __name__ == "__main__":
    
    pendulum_params = {
        'm': 1.0,
        'M': 5.0,
        'L': 1.0,
        'g': 9.81,
        'b': 0.1,
        'c': 0.01
    }
    
    initial_conditions = [
        0.0,
        0.0,
        np.radians(25.0),
        0.0
    ]
    
    t_simulation = 15.0
    
    colony_size = 30
    max_iterations = 30
    limit = 15
    
    bounds = [
        [10.0, 150.0],
        [0.0, 5.0],
        [5.0, 30.0],
        [0.0, 10.0],
        [0.0, 2.0],
        [0.0, 10.0]
    ]
    
    abc = ArtificialBeeColony(
        colony_size=colony_size,
        max_iterations=max_iterations,
        limit=limit,
        bounds=bounds
    )
    
    best_solution, best_fitness = abc.optimize(
        pendulum_params=pendulum_params,
        initial_conditions=initial_conditions,
        t_end=t_simulation,
        verbose=True
    )
    
    print("\nСоздание графиков сходимости...")
    fig = abc.plot_convergence()
    plt.savefig('abc_convergence.png', dpi=150, bbox_inches='tight')
    print("График сохранен: abc_convergence.png")
    
    pendulum = test_optimized_parameters(
        best_solution=best_solution,
        pendulum_params=pendulum_params,
        initial_conditions=initial_conditions,
        t_end=20.0
    )
    
    print("\nСохранение результатов...")
    
    with open('optimized_pid_parameters.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ОПТИМИЗИРОВАННЫЕ PID ПАРАМЕТРЫ (ABC ALGORITHM)\n")
        f.write("="*80 + "\n\n")
        f.write("Физические параметры маятника:\n")
        for key, value in pendulum_params.items():
            f.write(f"  {key} = {value}\n")
        f.write(f"\nНачальные условия:\n")
        f.write(f"  Угол = {np.degrees(initial_conditions[2]):.1f}°\n")
        f.write(f"\nНайденные PID параметры:\n")
        param_names = ['kp_angle', 'ki_angle', 'kd_angle', 'kp_pos', 'ki_pos', 'kd_pos']
        for i, name in enumerate(param_names):
            f.write(f"  {name} = {best_solution[i]:.4f}\n")
        f.write(f"\nЛучшая fitness: {best_fitness:.4f}\n")
        f.write(f"Количество итераций: {len(abc.history['best_fitness'])}\n")
        f.write("="*80 + "\n")
    
    print("Параметры сохранены: optimized_pid_parameters.txt")
    
    print("\n" + "="*80)
    print("ВСЕ ГОТОВО!")
    print("="*80)
    print("\nФайлы созданы:")
    print("  1. abc_convergence.png - графики сходимости алгоритма")
    print("  2. optimized_pid_parameters.txt - найденные параметры")
    print("\nВы можете использовать найденные параметры в вашей основной программе!")
    print("="*80)
    
    plt.show()