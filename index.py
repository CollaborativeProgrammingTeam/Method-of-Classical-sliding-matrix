
import numpy as np # импортируем библиотеку numpy для матричных вычислений.
from scipy.stats import f, pearsonr, t as t_dist # импорт статистические функции.
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков.

# Функция для дополнения матрицы независимых переменных X
def augment(X):
    """Дополняет матрицу X дополнительными степенями переменных."""
    N = len(X)
    augmented_X = np.ones((N, 5))  # Создаем матрицу размером Nx5 (5 переменных)
    augmented_X[:, 1] = X[:, 0]  # X1
    augmented_X[:, 2] = X[:, 0] ** 2  # X2 = X1^2
    augmented_X[:, 3] = X[:, 1]  # X3
    augmented_X[:, 4] = X[:, 0] * X[:, 1]  # X4 = X1 * X3
    return augmented_X

# Функция для выполнения регрессионного анализа данных и прогнозирования
def run_regression(X, Y):
    """
    Выполняет регрессионный анализ по данным из матрицы независимых переменных X и матрице-столбце зависимой переменной Y.
    Возвращает расчетные значения YR, коэффициенты регрессионной модели B и доверительные интервалы.
    
    Parameters:
    X (np.ndarray): Исходная матрица независимых переменных (Nx2).
    Y (np.ndarray): Исходный вектор зависимой переменной (Nx1).
    
    Returns:
    YR (np.ndarray): Расчетные значения Y.
    B (np.ndarray): Коэффициенты регрессии.
    Y_conf_low (np.ndarray): Нижняя граница доверительного интервала для YR.
    Y_conf_high (np.ndarray): Верхняя граница доверительного интервала для YR.
    """
    # 1. Формирование матрицы X с помощью augment
    augmented_X = augment(X)
    
    # 2. Расчет коэффициентов B регрессионной модели
    XT = augmented_X.T
    B = np.linalg.inv(XT @ augmented_X) @ XT @ Y
    
    # 3. Вычисление расчетных значений зависимой переменной YR
    YR = augmented_X @ B
    
    # 4. Проверка адекватности модели по F-критерию Фишера-Снедекора
    N, k = augmented_X.shape
    Dad = np.sum((Y - YR) ** 2) / (N - k)
    YSR = np.mean(Y)
    DY = np.sum((Y - YSR) ** 2) / (N - 1)
    FR = DY / Dad
    
    # Критическое значение F-статистики
    alpha = 0.05  # Уровень значимости альфа
    df1 = k - 1  # Число степеней свободы числителя
    df2 = N - k  # Число степеней свободы знаменателя
    F_critical = f.ppf(1 - alpha, df1, df2)  # Критическое значение F-статистики
    
    # Решение по адекватности модели
    decision = "Адекватна" if FR > F_critical else "Неадекватна"
    print(f"Модель {decision}. Коэффициент корреляции: {pearsonr(Y.flatten(), YR.flatten())[0]:.4f}, p-value: {pearsonr(Y.flatten(), YR.flatten())[1]:.4f}")
    
    # 5. Расчет доверительных интервалов для расчётных значений зависимой переменной YR
    G = np.linalg.inv(XT @ augmented_X)  # Ковариационная матрица коэффициентов
    
    # Число степеней свободы для t-критерия при оценке значимости коэффициентов
    df = N - k
    confidence = 0.95
    t_value = t_dist.ppf((1 + confidence) / 2, df)  # t-критерий для двустороннего интервала 95%
    
    # Стандартная ошибка прогнозирования YR
    # Используем диагональные элементы матрицы augmented_X @ G @ augmented_X.T
    SE_YR = np.sqrt(np.diag(augmented_X @ G @ augmented_X.T) * Dad)
    
    # Расчёт доверительных интервалов для расчётных значений зависимой переменной YR
    Y_conf_low = YR.flatten() - t_value * SE_YR
    Y_conf_high = YR.flatten() + t_value * SE_YR
    
    return YR.flatten(), B, Y_conf_low, Y_conf_high

# Исходные данные для первых 20 дней
raw_X_initial = np.array([
    [1, 21.5], [2, 21.2], [3, 22.1], [4, 25.1], [5, 26.4], [6, 22.6], [7, 17.7], [8, 18.5], [9, 21.2], [10, 20.3], 
    [11, 17], [12, 19.2], [13, 19.4], [14, 21.9], [15, 25.5], [16, 26.3], [17, 26.3], [18, 24.7], [19, 21.4], 
    [20, 21.04] ])
Y_initial = np.array([
    [2357.85], [2669.7], [2669.7], [2998.05], [3512.85], [3542.55], [3248.85], [3341.25], [3453.45],  
    [3598.65], [3413.85], [4271.85], [4393.95], [3686.1], [3682.8], [3550.8], [4719], [3979.35], [4131.6],   
    [4141.5] ])

# Фактические данные для прогнозирования на следующие дни (21-26)
additional_X = np.array([
    [21, 21.3],[22, 23],[23, 23.45],[24, 23.8],[25, 21.42],[26, 23.09] ])
additional_Y = np.array([ [4027.65],[3986.4],[3963.3],[4026],[3936.9],[3996.3]])

# Объединение данных
all_X = np.vstack((raw_X_initial, additional_X))
all_Y = np.vstack((Y_initial, additional_Y))

# Выполнение регрессии на исходных данных (20 дней)
YR_initial, B_initial, YR_initial_low, YR_initial_high = run_regression(raw_X_initial, Y_initial)

# Функция для выполнения скользящего окна и прогнозирования
def rolling_window_prediction(initial_X, initial_Y, additional_X, additional_Y, window_size=20):
    """
    Выполняет регрессионный анализ с использованием скользящего окна.
    Для каждой новой точки данных обновляет окно, выполняет регрессию и прогнозирует Y.

    Parameters:
    initial_X (np.ndarray): Исходная матрица X для начального окна.
    initial_Y (np.ndarray): Исходный вектор Y для начального окна.
    additional_X (np.ndarray): Дополнительные данные X для прогнозирования.
    additional_Y (np.ndarray): Дополнительные данные Y (фактические значения).
    window_size (int): Размер окна (по умолчанию 20 дней).

    Returns:
    predictions (list): Список прогнозируемых значений YR.
    predictions_low (list): Список нижних границ доверительных интервалов для прогнозов.
    predictions_high (list): Список верхних границ доверительных интервалов для прогнозов.
    actuals (list): Список фактических значений Y для дополнительных дней.
    days (list): Список номеров дней для дополнительных данных.
    """
    X_window = initial_X.copy()
    Y_window = initial_Y.copy()

    predictions = []
    predictions_low = []
    predictions_high = []
    actuals = []
    days = []

    # Проход по каждому новому дню
    for i in range(len(additional_X)):
        day_number = window_size + 1 + i
        temperature = additional_X[i][1]
        actual_Y = additional_Y[i][0]

        # Выполнение регрессионного анализа на текущем окне
        YR, B, YR_low, YR_high = run_regression(X_window, Y_window)

        # Формирование матрицы X для прогнозирования нового дня
        new_day_X = additional_X[i].reshape(1, -1)
        augmented_new_X = augment(new_day_X)

        # Прогнозирование Y для нового дня
        Y_pred = augmented_new_X @ B
        predicted_Y = Y_pred[0][0]  # Извлекаем скалярное значение

        # Доверительный интервал для прогнозных значений
        XT = augment(X_window).T
        G = np.linalg.inv(XT @ augment(X_window))
        Dad = np.sum((Y_window - (augment(X_window) @ B).reshape(-1, 1)) ** 2) / (len(Y_window) - augment(X_window).shape[1])
        SE_pred = np.sqrt(augmented_new_X @ G @ augmented_new_X.T * Dad)
        confidence = 0.95
        t_value = t_dist.ppf((1 + confidence) / 2, len(Y_window) - augment(X_window).shape[1])
        Y_pred_low = predicted_Y - t_value * SE_pred[0, 0]
        Y_pred_high = predicted_Y + t_value * SE_pred[0, 0]

        # Сохранение прогнозируемого и фактического значений
        predictions.append(predicted_Y)
        predictions_low.append(Y_pred_low)
        predictions_high.append(Y_pred_high)
        actuals.append(actual_Y)
        days.append(day_number)

        # Вывод только необходимых данных
        print(f"День {day_number}: Температура = {temperature}, Фактическое Y = {actual_Y}, Прогнозное Y = {predicted_Y:.2f}")

        # Обновление окна: удаление первого элемента и добавление нового - принцип метода скользящей матрицы
        X_window = np.vstack((X_window[1:], new_day_X))
        Y_window = np.vstack((Y_window[1:], [[actual_Y]]))

    return predictions, predictions_low, predictions_high, actuals, days

# Запуск функции скользящего окна и получение прогнозов
predicted_Y, predicted_Y_low, predicted_Y_high, actual_Y, prediction_days = rolling_window_prediction(
    raw_X_initial, Y_initial, additional_X, additional_Y
)

# Построение графика
plt.figure(figsize=(14, 8))

# 1. Фактические значения Y для всех 26 дней
plt.plot(all_X[:, 0], all_Y.flatten(), label='Фактическое электропотребление Y, кВт*ч.', marker='o', color='blue')

# 2. Расчетные значения YR для первых 20 дней
plt.plot(all_X[:20, 0], YR_initial, label='Расчетные значения зависимой переменной YR (дни 1-20)', marker='s', color='green')

# 3. Доверительный коридор для YR первых 20 дней
plt.fill_between(all_X[:20, 0], YR_initial_low, YR_initial_high,
                 color='green', alpha=0.2, label='Доверительный интервал коридора ошибок для расчётных значений зависимой переменной YR (1-20)')

# 4. Прогнозируемые значения Y для дней 21-26
plt.plot(prediction_days, predicted_Y, label='Прогнозные значения электропотребления (дни 21-26)', marker='x', linestyle='--', color='red')

# 5. Доверительный коридор интервала ошибок для прогнозных значений зависимой переменной Y
plt.fill_between(prediction_days, predicted_Y_low, predicted_Y_high,
                 color='red', alpha=0.2, label='Доверительный интервал коридора ошибок для прогнозных значений (21-26)')

plt.xlabel('День (сутки)')
plt.ylabel('Потребление электроэнергии (Y), кВт*ч.')
plt.title('Фактическое и прогнозируемое потребление электроэнергии по дням с доверительными интервалами')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 27))  # Устанавливаем метки по оси X для дней 1-26
plt.tight_layout()
plt.show()
