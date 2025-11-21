augment <- function(X) {
  N <- nrow(X)
  augmented_X <- matrix(1, nrow = N, ncol = 5)
  augmented_X[, 2] <- X[, 1]  # X1
  augmented_X[, 3] <- X[, 1]^2  # X2 = X1^2
  augmented_X[, 4] <- X[, 2]  # X3
  augmented_X[, 5] <- X[, 1] * X[, 2]  # X4 = X1 * X3
  return(augmented_X)
}

# Функция для выполнения регрессионного анализа данных и прогнозирования
run_regression <- function(X, Y) {
  # 1. Формирование матрицы X с помощью augment
  augmented_X <- augment(X)
  
  # 2. Расчет коэффициентов B регрессионной модели
  B <- solve(t(augmented_X) %*% augmented_X) %*% t(augmented_X) %*% Y
  
  # 3. Вычисление расчетных значений зависимой переменной YR
  YR <- augmented_X %*% B
  
  # 4. Проверка адекватности модели по F-критерию Фишера-Снедекора
  N <- nrow(augmented_X)
  k <- ncol(augmented_X)
  Dad <- sum((Y - YR)^2) / (N - k)
  YSR <- mean(Y)
  DY <- sum((Y - YSR)^2) / (N - 1)
  FR <- DY / Dad
  
  # Критическое значение F-статистики
  alpha <- 0.05
  df1 <- k - 1
  df2 <- N - k
  F_critical <- qf(1 - alpha, df1, df2)
  
  # Корреляция и p-value
  cor_test <- cor.test(as.vector(Y), as.vector(YR))
  
  # Решение по адекватности модели
  decision <- ifelse(FR > F_critical, "Адекватна", "Неадекватна")
  cat(sprintf("Модель %s. Коэффициент корреляции: %.4f, p-value: %.4f\n", 
              decision, cor_test$estimate, cor_test$p.value))
  
  # 5. Расчет доверительных интервалов
  G <- solve(t(augmented_X) %*% augmented_X)
  df <- N - k
  confidence <- 0.95
  t_value <- qt((1 + confidence) / 2, df)
  SE_YR <- sqrt(diag(augmented_X %*% G %*% t(augmented_X)) * Dad)
  Y_conf_low <- YR - t_value * SE_YR
  Y_conf_high <- YR + t_value * SE_YR
  
  return(list(
    YR = as.vector(YR),
    B = B,
    Y_conf_low = as.vector(Y_conf_low),
    Y_conf_high = as.vector(Y_conf_high)
  ))
}

# Функция для выполнения скользящего окна и прогнозирования
rolling_window_prediction <- function(initial_X, initial_Y, additional_X, additional_Y, window_size = 20) {
  X_window <- initial_X
  Y_window <- initial_Y

  predictions <- numeric()
  predictions_low <- numeric()
  predictions_high <- numeric()
  actuals <- numeric()
  days <- numeric()

  # Проход по каждому новому дню
  for (i in 1:nrow(additional_X)) {
    day_number <- window_size + i
    temperature <- additional_X[i, 2]
    actual_Y_val <- additional_Y[i, 1]

    # Выполнение регрессионного анализа на текущем окне
    result <- run_regression(X_window, Y_window)
    
    # Формирование матрицы X для прогнозирования нового дня
    new_day_X <- matrix(additional_X[i, ], nrow = 1)
    augmented_new_X <- augment(new_day_X)

    # Прогнозирование Y для нового дня
    Y_pred <- augmented_new_X %*% result$B
    predicted_Y <- as.numeric(Y_pred[1, 1])

    # Доверительный интервал для прогнозных значений
    XT_augmented <- augment(X_window)
    G <- solve(t(XT_augmented) %*% XT_augmented)
    Dad <- sum((Y_window - (XT_augmented %*% result$B))^2) / (nrow(Y_window) - ncol(XT_augmented))
    SE_pred <- sqrt(augmented_new_X %*% G %*% t(augmented_new_X) * Dad)
    confidence <- 0.95
    t_value <- qt((1 + confidence) / 2, nrow(Y_window) - ncol(XT_augmented))
    Y_pred_low <- predicted_Y - t_value * SE_pred[1, 1]
    Y_pred_high <- predicted_Y + t_value * SE_pred[1, 1]

    # Сохранение прогнозируемого и фактического значений
    predictions <- c(predictions, predicted_Y)
    predictions_low <- c(predictions_low, Y_pred_low)
    predictions_high <- c(predictions_high, Y_pred_high)
    actuals <- c(actuals, actual_Y_val)
    days <- c(days, day_number)

    # Вывод только необходимых данных
    cat(sprintf("День %d: Температура = %.2f, Фактическое Y = %.2f, Прогнозное Y = %.2f\n", 
                day_number, temperature, actual_Y_val, predicted_Y))

    # Обновление окна
    X_window <- rbind(X_window[-1, , drop = FALSE], new_day_X)
    Y_window <- rbind(Y_window[-1, , drop = FALSE], matrix(actual_Y_val, ncol = 1))
  }

  return(list(
    predictions = predictions,
    predictions_low = predictions_low,
    predictions_high = predictions_high,
    actuals = actuals,
    days = days
  ))
}

# Исходные данные для первых 20 дней
raw_X_initial <- matrix(c(
  1, 21.5,
  2, 21.2,
  3, 22.1,
  4, 25.1,
  5, 26.4,
  6, 22.6,
  7, 17.7,
  8, 18.5,
  9, 21.2,
  10, 20.3,
  11, 17,
  12, 19.2,
  13, 19.4,
  14, 21.9,
  15, 25.5,
  16, 26.3,
  17, 26.3,
  18, 24.7,
  19, 21.4,
  20, 21.04
), ncol = 2, byrow = TRUE)

Y_initial <- matrix(c(
  2357.85, 2669.7, 2669.7, 2998.05, 3512.85, 3542.55, 3248.85, 3341.25,
  3453.45, 3598.65, 3413.85, 4271.85, 4393.95, 3686.1, 3682.8, 3550.8,
  4719, 3979.35, 4131.6, 4141.5
), ncol = 1)

# Фактические данные для прогнозирования на следующие дни (21-26)
additional_X <- matrix(c(
  21, 21.3,
  22, 23,
  23, 23.45,
  24, 23.8,
  25, 21.42,
  26, 23.09
), ncol = 2, byrow = TRUE)

additional_Y <- matrix(c(
  4027.65,
  3986.4,
  3963.3,
  4026,
  3936.9,
  3996.3
), ncol = 1)

# Объединение данных
all_X <- rbind(raw_X_initial, additional_X)
all_Y <- rbind(Y_initial, additional_Y)

# Выполнение регрессии на исходных данных (20 дней)
cat("Регрессия на исходных данных (20 дней):\n")
result_initial <- run_regression(raw_X_initial, Y_initial)
YR_initial <- result_initial$YR
B_initial <- result_initial$B
YR_initial_low <- result_initial$Y_conf_low
YR_initial_high <- result_initial$Y_conf_high

# Запуск функции скользящего окна и получение прогнозов
cat("\nПрогнозирование с использованием скользящего окна:\n")
prediction_results <- rolling_window_prediction(
  raw_X_initial, Y_initial, additional_X, additional_Y
)

# Построение графика базовыми средствами R
par(mar = c(5, 4, 4, 8) + 0.1)  # Увеличиваем правый отступ для легенды

# Создаем основной график
plot(1:26, all_Y, type = "o", col = "blue", pch = 16, lwd = 2,
     xlab = "День (сутки)", ylab = "Потребление электроэнергии (Y), кВт*ч.",
     main = "Фактическое и прогнозируемое потребление электроэнергии по дням",
     xlim = c(1, 26), ylim = range(c(all_Y, YR_initial, prediction_results$predictions)),
     xaxt = "n")
axis(1, at = 1:26)

# Добавляем расчетные значения для первых 20 дней
lines(1:20, YR_initial, type = "o", col = "darkgreen", pch = 15, lwd = 2)

# Добавляем доверительный интервал для первых 20 дней
polygon(c(1:20, 20:1), 
        c(YR_initial_low, rev(YR_initial_high)), 
        col = rgb(0, 0.5, 0, 0.2), border = NA)

# Добавляем прогнозируемые значения для дней 21-26
lines(21:26, prediction_results$predictions, type = "o", 
      col = "red", pch = 4, lwd = 2, lty = 2)

# Добавляем доверительный интервал для прогнозов
polygon(c(21:26, 26:21), 
        c(prediction_results$predictions_low, rev(prediction_results$predictions_high)), 
        col = rgb(1, 0, 0, 0.2), border = NA)

# Добавляем легенду
legend("topright", 
       legend = c("Фактическое электропотребление Y, кВт*ч.",
                 "Расчетные значения YR (дни 1-20)",
                 "Прогнозные значения (дни 21-26)",
                 "Доверительный интервал YR (1-20)",
                 "Доверительный интервал прогноза (21-26)"),
       col = c("blue", "darkgreen", "red", 
               rgb(0, 0.5, 0, 0.3), rgb(1, 0, 0, 0.3)),
       pch = c(16, 15, 4, NA, NA),
       lty = c(1, 1, 2, 1, 1),
       lwd = c(2, 2, 2, 8, 8),
       pt.cex = c(1, 1, 1, 1, 1),
       inset = c(-0.3, 0),
       xpd = TRUE,
       bty = "n")

# Добавляем сетку
grid()

# Вывод коэффициентов регрессии
cat("\nКоэффициенты регрессионной модели для исходных данных:\n")
print(B_initial)

# Вывод статистики по прогнозам
cat("\nСтатистика прогнозов:\n")
for (i in 1:length(prediction_results$days)) {
  cat(sprintf("День %d: Прогноз = %.2f, Факт = %.2f, Ошибка = %.2f\n",
              prediction_results$days[i],
              prediction_results$predictions[i],
              prediction_results$actuals[i],
              prediction_results$predictions[i] - prediction_results$actuals[i]))
}
