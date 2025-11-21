package main
import (
    "fmt"   // Пакет для форматированного ввода-вывода (вывод результатов)
    "math"  // Математические функции (корень, логарифм, модуль и т.д.)
    "sort"  // Сортировка (используется в TInv для интерполяции)
)
// Matrix структура для работы с матрицами
// Используется для хранения данных и выполнения матричных операций
type Matrix struct {
    Rows, Cols int      // Размеры матрицы: количество строк и столбцов
    Data       []float64 // Элементы матрицы, хранящиеся в построчном порядке
}
// NewMatrix создает новую матрицу с проверкой корректности размера данных
// rows - количество строк, cols - количество столбцов
// data - элементы матрицы в виде одномерного слайса
func NewMatrix(rows, cols int, data []float64) Matrix {
    if len(data) != rows*cols {
        panic("Неверный размер данных для матрицы")
    }
    return Matrix{Rows: rows, Cols: cols, Data: data}
}
// At получает элемент матрицы по индексам (строка i, столбец j)
// Индексация начинается с 0
func (m Matrix) At(i, j int) float64 {
    return m.Data[i*m.Cols+j]
}
// Set устанавливает значение элемента матрицы по индексам
func (m Matrix) Set(i, j int, value float64) {
    m.Data[i*m.Cols+j] = value
}
// Multiply умножает две матрицы: A (m×n) * B (n×p) = C (m×p)
// Требование: количество столбцов A должно равняться количеству строк B
func Multiply(a, b Matrix) Matrix {
    if a.Cols != b.Rows {
        panic("Несовместимые размеры матриц для умножения")
    }
    result := NewMatrix(a.Rows, b.Cols, make([]float64, a.Rows*b.Cols))
    for i := 0; i < a.Rows; i++ {
        for j := 0; j < b.Cols; j++ {
            sum := 0.0
            for k := 0; k < a.Cols; k++ {
                sum += a.At(i, k) * b.At(k, j)
            }
            result.Set(i, j, sum)
        }
    }
    return result
}
// Transpose возвращает транспонированную матрицу
// Строки становятся столбцами, столбцы - строками
func Transpose(m Matrix) Matrix {
    result := NewMatrix(m.Cols, m.Rows, make([]float64, m.Rows*m.Cols))
    for i := 0; i < m.Rows; i++ {
        for j := 0; j < m.Cols; j++ {
            result.Set(j, i, m.At(i, j))
        }
    }
    return result
}
// Inverse возвращает обратную матрицу методом Гаусса-Жордана
// Работает только для квадратных невырожденных матриц
func Inverse(m Matrix) Matrix {
    if m.Rows != m.Cols {
        panic("Матрица должна быть квадратной для обращения")
    }

    n := m.Rows
    // Создаем расширенную матрицу [A|I], где I - единичная матрица
    augmented := NewMatrix(n, 2*n, make([]float64, n*2*n))

    // Заполняем левую часть исходной матрицей, правую - единичной
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            augmented.Set(i, j, m.At(i, j))
        }
        augmented.Set(i, i+n, 1.0)
    }

    // Прямой ход метода Гаусса-Жордана
    for i := 0; i < n; i++ {
        // Поиск главного элемента (максимального по модулю в столбце)
        maxRow := i
        for k := i + 1; k < n; k++ {
            if math.Abs(augmented.At(k, i)) > math.Abs(augmented.At(maxRow, i)) {
                maxRow = k
            }
        }

        // Перестановка строк для обеспечения устойчивости алгоритма
        if maxRow != i {
            for j := 0; j < 2*n; j++ {
                temp := augmented.At(i, j)
                augmented.Set(i, j, augmented.At(maxRow, j))
                augmented.Set(maxRow, j, temp)
            }
        }

        // Нормализация текущей строки (деление на ведущий элемент)
        pivot := augmented.At(i, i)
        if math.Abs(pivot) < 1e-10 {
            panic("Матрица вырождена")
        }

        for j := 0; j < 2*n; j++ {
            augmented.Set(i, j, augmented.At(i, j)/pivot)
        }

        // Исключение - обнуление элементов в текущем столбце других строк
        for k := 0; k < n; k++ {
            if k != i {
                factor := augmented.At(k, i)
                for j := 0; j < 2*n; j++ {
                    augmented.Set(k, j, augmented.At(k, j)-factor*augmented.At(i, j))
                }
            }
        }
    }

    // Извлекаем обратную матрицу из правой части расширенной матрицы
    result := NewMatrix(n, n, make([]float64, n*n))
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            result.Set(i, j, augmented.At(i, j+n))
        }
    }

    return result
}

// Mean вычисляет среднее арифметическое значение массива
func Mean(data []float64) float64 {
    sum := 0.0
    for _, v := range data {
        sum += v
    }
    return sum / float64(len(data))
}

// Augment дополняет матрицу независимых переменных для полиномиальной регрессии 2-го порядка
// Преобразует матрицу X [день, температура] в расширенную матрицу с признаками:
// [1, X1, X1², X3, X1*X3] где X1 - номер дня, X3 - температура
func Augment(X Matrix) Matrix {
    N := X.Rows
    augmentedData := make([]float64, N*5)

    for i := 0; i < N; i++ {
        augmentedData[i*5+0] = 1                          // Константа (свободный член модели)
        augmentedData[i*5+1] = X.At(i, 0)                 // X1 (номер дня - линейный эффект)
        augmentedData[i*5+2] = X.At(i, 0) * X.At(i, 0)   // X1² (квадрат номера дня - нелинейный эффект)
        augmentedData[i*5+3] = X.At(i, 1)                 // X3 (температура - линейный эффект)
        augmentedData[i*5+4] = X.At(i, 0) * X.At(i, 1)   // X1*X3 (взаимодействие дня и температуры)
    }

    return NewMatrix(N, 5, augmentedData)
}

// RegressionResult содержит полные результаты регрессионного анализа
type RegressionResult struct {
    YR[] float64 // Расчетные значения зависимой переменной Y
    B  Matrix // Коэффициенты регрессии [B0, B1, B2, B3, B4]
    YConfLow []float64 // Нижние границы 95% доверительных интервалов для Y
    YConfHigh []float64 // Верхние границы 95% доверительных интервалов для Y
    Correlation float64 // Коэффициент корреляции между Y и YR
    Decision string // Решение об адекватности модели ("Адекватна"/"Неадекватна")
}

// RunRegression выполняет полный регрессионный анализ по методу наименьших квадратов
// Возвращает коэффициенты модели, прогнозы и статистики качества
func RunRegression(X, Y Matrix) RegressionResult {
    // 1. Расширение матрицы признаков для полиномиальной регрессии
    augmentedX := Augment(X)

    // 2. Расчет коэффициентов регрессии: B = (XᵀX)⁻¹XᵀY
    XT := Transpose(augmentedX)
    XTX := Multiply(XT, augmentedX)
    XTXInv := Inverse(XTX)
    XTY := Multiply(XT, Y)
    B := Multiply(XTXInv, XTY)

    // 3. Расчет прогнозных значений YR = X * B
    YRMatrix := Multiply(augmentedX, B)
    YR := make([]float64, Y.Rows)
    for i := 0; i < Y.Rows; i++ {
        YR[i] = YRMatrix.At(i, 0)
    }

    // 4. Проверка адекватности модели по F-критерию Фишера
    N := augmentedX.Rows // Количество наблюдений
    k := augmentedX.Cols // Количество параметров модели (5)

    // Дисперсия адекватности (остаточная дисперсия)
    sumSquaredErrors := 0.0
    for i := 0; i < N; i++ {
        error := Y.At(i, 0) - YR[i]
        sumSquaredErrors += error * error
    }
    Dad := sumSquaredErrors / float64(N-k)

    // Общая дисперсия зависимой переменной
    YSR := Mean(Y.Data)
    totalSumSquares := 0.0
    for i := 0; i < N; i++ {
        totalSumSquares += (Y.At(i, 0) - YSR) * (Y.At(i, 0) - YSR)
    }
    DY := totalSumSquares / float64(N-1)

    // F-статистика: отношение объясненной дисперсии к остаточной
    FR := DY / Dad

    // Критическое значение F-распределения для уровня значимости 5%
    alpha := 0.05
    df1 := k - 1  // Степени свободы числителя
    df2 := N - k  // Степени свободы знаменателя
    Fcritical := FInv(alpha, df1, df2)

    // Коэффициент корреляции между фактическими и расчетными значениями
    correlation := Correlation(Y.Data, YR)

    // Проверка адекватности: если F > Fкрит, модель адекватна
    decision := "Неадекватна"
    if FR > Fcritical {
        decision = "Адекватна"
    }

    fmt.Printf("Модель %s. Коэффициент корреляции: %.4f\n", decision, correlation)

    // 5. Расчет доверительных интервалов для прогнозных значений
    G := XTXInv // Матрица ковариаций коэффициентов (XᵀX)⁻¹
    df := N - k // Степени свободы
    confidence := 0.95
    tValue := TInv((1+confidence)/2, df) // Критическое значение t-статистики

    YConfLow := make([]float64, N)
    YConfHigh := make([]float64, N)

    for i := 0; i < N; i++ {
       
 // Вектор признаков для i-го наблюдения
        xi := make([]float64, 5)
        for j := 0; j < 5; j++ {
            xi[j] = augmentedX.At(i, j)
        }

        // Дисперсия прогноза: Var(ŷ) = σ² * xᵢ(XᵀX)⁻¹xᵢᵀ
        var seSquared float64
        for j := 0; j < 5; j++ {
            for l := 0; l < 5; l++ {
                seSquared += xi[j] * G.At(j, l) * xi[l]
            }
        }
        SE_YR := math.Sqrt(seSquared * Dad) // Стандартная ошибка прогноза

        // Доверительный интервал: ŷ ± t(α/2, df) * SE(ŷ)
        YConfLow[i] = YR[i] - tValue*SE_YR
        YConfHigh[i] = YR[i] + tValue*SE_YR
    }

    return RegressionResult{
        YR:          YR,
        B:           B,
        YConfLow:    YConfLow,
        YConfHigh:   YConfHigh,
        Correlation: correlation,
        Decision:    decision,
    }
}

// PredictionResult содержит результаты прогнозирования на новых данных
type PredictionResult struct {
    Predictions     []float64 // Точечные прогнозы для новых наблюдений
    PredictionsLow  []float64 // Нижние границы доверительных интервалов
    PredictionsHigh []float64 // Верхние границы доверительных интервалов
    Actuals         []float64 // Фактические значения для проверки точности
    Days            []int     // Номера дней, для которых сделан прогноз
}

// RollingWindowPrediction реализует прогнозирование с скользящим окном
// На каждом шаге добавляет новые данные, удаляет старые и перестраивает модель
// windowSize - размер окна (20 дней в данном случае)
func RollingWindowPrediction(initialX, initialY, additionalX, additionalY Matrix, windowSize int) PredictionResult {
    XWindow := initialX // Текущее окно признаков
    YWindow := initialY // Текущее окно целевых значений

    predictions := make([]float64, 0)
    predictionsLow := make([]float64, 0)
    predictionsHigh := make([]float64, 0)
    actuals := make([]float64, 0)
    days := make([]int, 0)

    // Последовательная обработка каждого нового дня
    for i := 0; i < additionalX.Rows; i++ {
        dayNumber := windowSize + i + 1  // Номер текущего дня (21, 22, ...)
        temperature := additionalX.At(i, 1)
        actualYVal := additionalY.At(i, 0)

        // Обучение модели на текущем скользящем окне
        result := RunRegression(XWindow, YWindow)

        // Подготовка данных нового дня для прогноза
        newDayX := NewMatrix(1, 2, []float64{
            additionalX.At(i, 0), additionalX.At(i, 1),
        })
        augmentedNewX := Augment(newDayX)

        // Точечный прогноз: ŷ = X_new * B
        YPredMatrix := Multiply(augmentedNewX, result.B)
        predictedY := YPredMatrix.At(0, 0)

        // Расчет доверительного интервала для прогноза нового наблюдения
        XTaugmented := Augment(XWindow)
        G := Inverse(Multiply(Transpose(XTaugmented), XTaugmented))

        // Оценка дисперсии ошибки на текущем окне
        sumSquaredErrors := 0.0
        for j := 0; j < YWindow.Rows; j++ {
            YRj := Multiply(XTaugmented, result.B).At(j, 0)
            error := YWindow.At(j, 0) - YRj
            sumSquaredErrors += error * error
        }
        Dad := sumSquaredErrors / float64(YWindow.Rows-XTaugmented.Cols)

        // Стандартная ошибка прогноза для нового наблюдения
        xi := make([]float64, 5)
        for j := 0; j < 5; j++ {
            xi[j] = augmentedNewX.At(0, j)
        }

        var seSquared float64
        for j := 0; j < 5; j++ {
            for l := 0; l < 5; l++ {
                seSquared += xi[j] * G.At(j, l) * xi[l]
            }
        }
        SEPred := math.Sqrt(seSquared * Dad)

        confidence := 0.95
        tValue := TInv((1+confidence)/2, YWindow.Rows-XTaugmented.Cols)

        // Границы доверительного интервала прогноза
        YPredLow := predictedY - tValue*SEPred
        YPredHigh := predictedY + tValue*SEPred

        // Сохранение результатов прогноза
        predictions = append(predictions, predictedY)
        predictionsLow = append(predictionsLow, YPredLow)
        predictionsHigh = append(predictionsHigh, YPredHigh)
        actuals = append(actuals, actualYVal)
        days = append(days, dayNumber)

        fmt.Printf("День %d: Температура = %.2f, Фактическое Y = %.2f, Прогнозное Y = %.2f\n",
            dayNumber, temperature, actualYVal, predictedY)

        // Обновление скользящего окна: удаление самого старого наблюдения,
        // добавление нового (принцип FIFO - First In First Out)
        newXData := make([]float64, (XWindow.Rows)*XWindow.Cols)
        newYData := make([]float64, (YWindow.Rows)*YWindow.Cols)

        // Копируем все строки кроме первой (удаляем самую старую)
        for j := 1; j < XWindow.Rows; j++ {
            for c := 0; c < XWindow.Cols; c++ {
                newXData[(j-1)*XWindow.Cols+c] = XWindow.At(j, c)
            }
        }
        // Добавляем новую строку в конец
        for c := 0; c < XWindow.Cols; c++ {
            newXData[(XWindow.Rows-1)*XWindow.Cols+c] = newDayX.At(0, c)
        }

        // Аналогично для целевых значений
        for j := 1; j < YWindow.Rows; j++ {
            newYData[(j-1)*YWindow.Cols] = YWindow.At(j, 0)
        }
        newYData[(YWindow.Rows-1)*YWindow.Cols] = actualYVal

        XWindow = NewMatrix(XWindow.Rows, XWindow.Cols, newXData)
        YWindow = NewMatrix(YWindow.Rows, YWindow.Cols, newYData)
    }

    return PredictionResult{
        Predictions:     predictions,
        PredictionsLow:  predictionsLow,
        PredictionsHigh: predictionsHigh,
        Actuals:         actuals,
        Days:            days,
    }
}

// Correlation вычисляет коэффициент корреляции Пирсона между двумя выборками
// Возвращает значение от -1 до 1, где:
//  1 - полная положительная корреляция
// -1 - полная отрицательная корреляция  
//  0 - отсутствие линейной связи
func Correlation(x, y []float64) float64 {
    if len(x) != len(y) {
        panic("Размеры массивов должны совпадать")
    }

    n := len(x)
    sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0

    // Вычисление необходимых сумм для формулы корреляции Пирсона
    for i := 0; i < n; i++ {
        sumX += x[i]
        sumY += y[i]
        sumXY += x[i] * y[i]
        sumX2 += x[i] * x[i]
        sumY2 += y[i] * y[i]
    }

    // Формула корреляции Пирсона:
    // r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))
    numerator := float64(n)*sumXY - sumX*sumY
    denominator := math.Sqrt((float64(n)*sumX2 - sumX*sumX) * (float64(n)*sumY2 - sumY*sumY))

    if denominator == 0 {
        return 0 // Избегаем деления на ноль
    }
    return numerator / denominator
}

// TInv вычисляет критическое значение t-распределения Стьюдента
// probability - доверительная вероятность (например, 0.975 для 95% ДИ)
// df - степени свободы
func TInv(probability float64, df int) float64 {
    if df <= 0 {
        panic("Степени свободы должны быть положительными")
    }

    // Для больших степеней свободы (>30) приближаем нормальным распределением
    if df > 30 {
        return NormalInv(probability)
    }

    // Таблица критических значений t-распределения для различных df
    tTable := map[int]map[float64]float64{
        1:  {0.95: 6.314, 0.975: 12.706},
        2:  {0.95: 2.920, 0.975: 4.303},
        3:  {0.95: 2.353, 0.975: 3.182},
        4:  {0.95: 2.132, 0.975: 2.776},
        5:  {0.95: 2.015, 0.975: 2.571},
        6:  {0.95: 1.943, 0.975: 2.447},
        7:  {0.95: 1.895, 0.975: 2.365},
        8:  {0.95: 1.860, 0.975: 2.306},
        9:  {0.95: 1.833, 0.975: 2.262},
        10: {0.95: 1.812, 0.975: 2.228},
        15: {0.95: 1.753, 0.975: 2.131},
        20: {0.95: 1.725, 0.975: 2.086},
        25: {0.95: 1.708, 0.975: 2.060},
        30: {0.95: 1.697, 0.975: 2.042},
    }

    // Поиск значения в таблице
    if dfTable, exists := tTable[df]; exists {
        if value, exists := dfTable[probability]; exists {
            return value
        }
    }

    // Линейная интерполяция для отсутствующих в таблице значений df
    keys := make([]int, 0, len(tTable))
    for k := range tTable {
        keys = append(keys, k)
    }
    sort.Ints(keys) // Сортируем ключи для интерполяции

    for i := 0; i < len(keys)-1; i++ {
        if df >= keys[i] && df <= keys[i+1] {
            lower := tTable[keys[i]][probability]
            upper := tTable[keys[i+1]][probability]
            weight := float64(df-keys[i]) / float64(keys[i+1]-keys[i])
            return lower + weight*(upper-lower)
        }
    }

    // Если df больше максимального в таблице, используем нормальное приближение
    return NormalInv(probability)
}

// FInv вычисляет критическое значение F-распределения Фишера
// alpha - уровень значимости (0.05 для 95% доверительной вероятности)
// df1, df2 - степени свободы числителя и знаменателя
func FInv(alpha float64, df1, df2 int) float64 {
    // Таблица критических значений F-распределения для α=0.05
    fTable := map[string]map[int]map[int]float64{
        "0.05": {
            1: {
                1: 161.4, 2: 18.51, 3: 10.13, 4: 7.71, 5: 6.61,
                10: 4.96, 20: 4.35, 30: 4.17,
            },
            2: {
                1: 199.5, 2: 19.00, 3: 9.55, 4: 6.94, 5: 5.79,
                10: 4.10, 20: 3.49, 30: 3.32,
            },
            3: {
                1: 215.7, 2: 19.16, 3: 9.28, 4: 6.59, 5: 5.41,
                10: 3.71, 20: 3.10, 30: 2.92,
            },
            4: {
                1: 224.6, 2: 19.25, 3: 9.12, 4: 6.39, 5: 5.19,
                10: 3.48, 20: 2.87, 30: 2.69,
            },
            5: {
                1: 230.2, 2: 19.30, 3: 9.01, 4: 6.26, 5: 5.05,
                10: 3.33, 20: 2.71, 30: 2.53,
            },
        },
    }

    // Поиск значения в таблице
    if alphaTable, exists := fTable[fmt.Sprintf("%.2f", alpha)]; exists {
        if df1Table, exists := alphaTable[df1]; exists {
            if value, exists := df1Table[df2]; exists {
                return value
            }
        }
    }

    // Упрощенное значение при отсутствии в таблице
    return 3.0
}

// NormalInv вычисляет квантиль стандартного нормального распределения
// Используется аппроксимация Пэка для обратной функции нормального распределения
// p - вероятность (должна быть в интервале (0, 1))
func NormalInv(p float64) float64 {
    if p <= 0 || p >= 1 {
        panic("Вероятность должна быть в интервале (0, 1)")
    }

    // Для вероятностей меньше 0.5 используем симметричность распределения
    if p < 0.5 {
        return -NormalInv(1 - p)
    }

    // Аппроксимация Пэка для p >= 0.5
    t := math.Sqrt(-2 * math.Log(1-p))
    c0 := 2.515517
    c1 := 0.802853
    c2 := 0.010328
    d1 := 1.432788
    d2 := 0.189269
    d3 := 0.001308

    return t - (c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t)
}

func main() {
    // Исходные данные для обучения модели (первые 20 дней)
    // Матрица X: каждая строка содержит [номер дня, температура]
    rawXInitial := NewMatrix(20, 2, []float64{
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
        20, 21.04,
    })

    // Вектор Y: потребление электроэнергии за каждый день (кВт*ч)
    YInitial := NewMatrix(20, 1, []float64{
        2357.85, 2669.7, 2669.7, 2998.05, 3512.85, 3542.55, 3248.85, 3341.25,
        3453.45, 3598.65, 3413.85, 4271.85, 4393.95, 3686.1, 3682.8, 3550.8,
        4719, 3979.35, 4131.6, 4141.5,
    })

    // Данные для прогнозирования (дни 21-26)
    additionalX := NewMatrix(6, 2, []float64{
        21, 21.3,
        22, 23,
        23, 23.45,
        24, 23.8,
        25, 21.42,
        26, 23.09,
    })

    // Фактические значения для проверки точности прогнозов
    additionalY := NewMatrix(6, 1, []float64{
        4027.65,
        3986.4,
        3963.3,
        4026,
        3936.9,
        3996.3,
    })

    // Объединение данных для отображения полной картины в результатах
    allYData := append(YInitial.Data, additionalY.Data...)
    allY := NewMatrix(26, 1, allYData)

    // Регрессионный анализ на исходных данных (20 дней)
    fmt.Println("Регрессия на исходных данных (20 дней):")
    resultInitial := RunRegression(rawXInitial, YInitial)

    // Прогнозирование с обновлением модели по скользящему окну
    fmt.Println("\nПрогнозирование с использованием скользящего окна:")
    predictionResults := RollingWindowPrediction(
        rawXInitial, YInitial, additionalX, additionalY, 20,
    )

    // Вывод коэффициентов регрессионной модели
    fmt.Println("\nКоэффициенты регрессионной модели для исходных данных:")
    for i := 0; i < resultInitial.B.Rows; i++ {
        fmt.Printf("B%d = %.4f\n", i, resultInitial.B.At(i, 0))
    }

    // Статистика точности прогнозов для дней 21-26
    fmt.Println("\nСтатистика прогнозов:")
    for i := 0; i < len(predictionResults.Days); i++ {
        error := predictionResults.Predictions[i] - predictionResults.Actuals[i]
        fmt.Printf("День %d: Прогноз = %.2f, Факт = %.2f, Ошибка = %.2f\n",
            predictionResults.Days[i],
            predictionResults.Predictions[i],
            predictionResults.Actuals[i],
            error)
    }
    // Сводная таблица всех результатов (обучающая выборка + прогнозы)
    fmt.Println("\nТекстовая визуализация результатов:")
    fmt.Println("День | Факт Y | Расчет YR | Прогноз | Дов.интер.Min | Дов.интер.Max ")
    fmt.Println("-----|--------|-----------|---------|--------|--------")

    // Результаты для дней 1-20 (обучающая выборка)
    for i := 0; i < 20; i++ {
        fmt.Printf("%4d | %6.1f | %9.1f | %7s | %6.1f | %6.1f\n",
            i+1, allY.At(i, 0), resultInitial.YR[i], "-",
            resultInitial.YConfLow[i], resultInitial.YConfHigh[i])
    }
    // Результаты для дней 21-26 (прогноз с обновлением модели)
    for i := 0; i < len(predictionResults.Days); i++ {
        fmt.Printf("%4d | %6.1f | %9s | %7.1f | %6.1f | %6.1f\n",
            predictionResults.Days[i], predictionResults.Actuals[i], "-",
            predictionResults.Predictions[i],
            predictionResults.PredictionsLow[i],
            predictionResults.PredictionsHigh[i])
    }
}
