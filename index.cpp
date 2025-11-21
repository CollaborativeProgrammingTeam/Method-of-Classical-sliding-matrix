Основные структуры данных и вспомогательные функции
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

using namespace std;

// Создаем псевдонимы для удобства работы с векторами и матрицами
typedef vector<double> Vec;    // Вектор чисел
typedef vector<Vec> Mat;       // Матрица (вектор векторов)

// Функция для красивого вывода вектора
void printVec(const Vec& v, const string& name) {
    cout << name << ": ";
    for(double val : v) cout << val << " ";
    cout << endl;
}
Математические операции с матрицами
Умножение матриц
Mat matMul(const Mat& A, const Mat& B) {
    int m = A.size();     // Количество строк в A
    int n = A[0].size();  // Количество столбцов в A
    int p = B[0].size();  // Количество столбцов в B
    
    Mat C(m, Vec(p, 0.0)); // Создаем результирующую матрицу
    
    // Классическое умножение матриц: C[i][j] = сумма по k от A[i][k]*B[k][j]
    for(int i=0; i<m; ++i)
        for(int j=0; j<p; ++j)
            for(int k=0; k<n; ++k)
                C[i][j] += A[i][k]*B[k][j];
    return C;
}
Транспонирование матрицы
Mat transpose(const Mat& A) {
    int m = A.size();     // Количество строк
    int n = A[0].size();  // Количество столбцов
    Mat AT(n, Vec(m));    // Создаем транспонированную матрицу (размеры меняются местами)
    
    // Элемент [i][j] становится элементом [j][i]
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j)
            AT[j][i] = A[i][j];
    return AT;
}
Обращение матрицы (метод Гаусса-Жордана)
Mat inverse(const Mat& A) {
    int n = A.size();
    // Создаем расширенную матрицу [A|I], где I - единичная матрица
    Mat B(n, Vec(n*2, 0.0));
    
    // Заполняем левую часть исходной матрицей A, правую - единичной матрицей
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j)
            B[i][j] = A[i][j];
        B[i][i+n] = 1.0;  // Диагональные элементы правой части = 1
    }
    
    // Прямой ход метода Гаусса
    for(int i=0; i<n; ++i){
        // Поиск главного элемента (максимального по модулю в столбце)
        double maxEl = abs(B[i][i]);
        int maxRow = i;
        for(int k=i+1; k<n; ++k){
            if(abs(B[k][i]) > maxEl){
                maxEl = abs(B[k][i]);
                maxRow = k;
            }
        }
        
        // Перестановка строк, если нужно
        if(maxRow != i) swap(B[i], B[maxRow]);
        
        // Проверка на вырожденность матрицы
        if(abs(B[i][i]) < 1e-15) throw runtime_error("Matrix is singular!");

        // Нормировка текущей строки (делаем диагональный элемент = 1)
        double diagEl = B[i][i];
        for(int j=0; j<2*n; ++j)
            B[i][j] /= diagEl;

        // Обнуление элементов в текущем столбце для других строк
        for(int k=0; k<n; ++k){
            if(k == i) continue;
            double coeff = B[k][i];
            for(int j=0; j<2*n; ++j){
                B[k][j] -= coeff * B[i][j];
            }
        }
    }
    
    // Извлекаем обратную матрицу из правой части расширенной матрицы
    Mat inv(n, Vec(n));
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            inv[i][j] = B[i][j+n];
    return inv;
}
Умножение матрицы на вектор
Vec matVecMul(const Mat& A, const Vec& x) {
    int m = A.size();     // Количество строк в A
    int n = A[0].size();  // Количество столбцов в A
    
    if((int)x.size() != n) throw runtime_error("matVecMul size mismatch");
    
    Vec y(m, 0.0);  // Результирующий вектор
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j)
            y[i] += A[i][j]*x[j];
    return y;
}
Расширение матрицы признаков для полиномиальной регрессии
Mat augment(const Mat& X) {
    int N = X.size();  // Количество наблюдений
    
    // Создаем расширенную матрицу с 5 признаками:
    // [1, x1, x1², x2, x1*x2] - полином второй степени с взаимодействием
    Mat augmented_X(N, Vec(5,1.0));
    
    for(int i=0; i<N; ++i) {
        augmented_X[i][1] = X[i][0];        // x1 (первый признак)
        augmented_X[i][2] = X[i][0]*X[i][0]; // x1² (квадрат первого признака)
        augmented_X[i][3] = X[i][1];        // x2 (второй признак)
        augmented_X[i][4] = X[i][0]*X[i][1]; // x1*x2 (взаимодействие)
    }
    return augmented_X;
}
Статистические функции
Среднее значение
double mean(const Vec& v) {
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}
Коэффициент корреляции Пирсона
double pearsonCoeff(const Vec& y, const Vec& y_pred) {
    double mean_y = mean(y);     // Среднее фактических значений
    double mean_yp = mean(y_pred); // Среднее прогнозных значений
    
    double cov = 0;    // Ковариация
    double var_y = 0;  // Дисперсия y
    double var_yp = 0; // Дисперсия y_pred
    
    for(size_t i=0; i<y.size(); ++i) {
        cov += (y[i]-mean_y)*(y_pred[i]-mean_yp);
        var_y += (y[i]-mean_y)*(y[i]-mean_y);
        var_yp += (y_pred[i]-mean_yp)*(y_pred[i]-mean_yp);
    }
    
    if(var_y == 0 || var_yp == 0) return 0.0;
    return cov/std::sqrt(var_y*var_yp);  // Коэффициент корреляции
}
F-статистика для проверки адекватности модели
double fStatistic(const Vec& y, const Vec& y_pred, int N, int k){
    // Dad - дисперсия ошибок (остаточная дисперсия)
    double Dad = 0;
    for(int i=0; i<N; ++i) {
        Dad += (y[i] - y_pred[i])*(y[i] - y_pred[i]);
    }
    Dad /= (N - k);  // N-k степеней свободы

    // DY - общая дисперсия зависимой переменной
    double YSR = mean(y);
    double DY = 0;
    for(int i=0; i<N; ++i) {
        DY += (y[i]-YSR)*(y[i]-YSR);
    }
    DY /= (N - 1);  // N-1 степеней свободы

    return DY/Dad;  // F-статистика
}
Критическое значение t-распределения для 95% доверительного интервала
double tValue95(int df){
    // Приближенные значения t-статистики для разных степеней свободы
    if(df > 30) return 2.04;
    else if(df > 20) return 2.09;
    else if(df > 10) return 2.23;
    else return 2.35;
}
Основная логика регрессионного анализа
struct RegressionResult {
    Vec YR;           // Прогнозные значения
    Vec B;            // Коэффициенты регрессии
    Vec Y_conf_low;   // Нижняя граница доверительного интервала
    Vec Y_conf_high;  // Верхняя граница доверительного интервала
};

RegressionResult run_regression(const Mat& X, const Vec& Y) {
    // 1. Расширяем матрицу признаков
    Mat augmented_X = augment(X);
    int N = augmented_X.size();  // Количество наблюдений
    int k = augmented_X[0].size(); // Количество коэффициентов

    // 2. Вычисляем XT * X (матрицу моментов)
    Mat XT = transpose(augmented_X);
    Mat XT_X = matMul(XT, augmented_X);
    Mat XT_X_inv = inverse(XT_X);  // Обращаем матрицу

    // 3. Вычисляем XT * Y
    Vec XT_Y(k, 0.0);
    for(int i=0; i<k; ++i)
        for(int j=0; j<N; ++j)
            XT_Y[i] += XT[i][j]*Y[j];

    // 4. Вычисляем коэффициенты регрессии: B = (XT*X)^(-1) * XT * Y
    Vec B = matVecMul(XT_X_inv, XT_Y);

    // 5. Вычисляем прогнозные значения: YR = X * B
    Vec YR(N, 0.0);
    for(int i=0; i<N; ++i)
        for(int j=0; j<k; ++j)
            YR[i] += augmented_X[i][j] * B[j];

    // 6. Вычисляем статистики качества модели
    double FR = fStatistic(Y, YR, N, k);
    cout << "Статистика F (адекватность): " << FR << endl;

    double r = pearsonCoeff(Y, YR);
    cout << "Коэффициент корреляции Пирсона: " << r << endl;

    // 7. Вычисляем доверительные интервалы для прогнозов
    double Dad = 0;
    for(int i=0; i<N; ++i)
        Dad += (Y[i]-YR[i])*(Y[i]-YR[i]);
    Dad /= (N-k);  // Остаточная дисперсия

    // Стандартные ошибки прогнозов
    Vec SE_YR(N, 0.0);
    for(int i=0; i<N; ++i) {
        double s = 0;
        for(int j=0; j<k; ++j)
            for(int m=0; m<k; ++m)
                s += augmented_X[i][j] * XT_X_inv[j][m] * augmented_X[i][m];
        SE_YR[i] = sqrt(s * Dad);
    }

    // 8. Вычисляем доверительные интервалы
    int df = N-k;
    double t_val = tValue95(df);

    Vec Y_conf_low(N), Y_conf_high(N);
    for(int i=0; i<N; ++i) {
        Y_conf_low[i] = YR[i] - t_val*SE_YR[i];
        Y_conf_high[i] = YR[i] + t_val*SE_YR[i];
    }

    return {YR, B, Y_conf_low, Y_conf_high};
}

Прогнозирование с использованием скользящей матрицы
struct Prediction {
    vector<double> pred;       // Прогнозы
    vector<double> pred_low;   // Нижние границы доверительных интервалов
    vector<double> pred_high;  // Верхние границы доверительных интервалов
    vector<double> actual;     // Фактические значения
    vector<int> days;          // Номера дней
};

Prediction rolling_window_prediction(const Mat& initial_X, const Vec& initial_Y,
                                     const Mat& additional_X, const Vec& additional_Y,
                                     int window_size = 20) {
    // Начинаем с исходного окна данных
    Mat X_window = initial_X;
    Vec Y_window = initial_Y;

    vector<double> predictions, predictions_low, predictions_high, actuals;
    vector<int> days;

    // Для каждого нового дня делаем прогноз
    for(size_t i=0; i<additional_X.size(); ++i) {
        int day_num = window_size + i + 1;

        // 1. Строим регрессионную модель на текущем окне
        RegressionResult res = run_regression(X_window, Y_window);

        // 2. Подготавливаем данные для прогноза на следующий день
        Mat new_day = {additional_X[i]};
        Mat aug_new_day = augment(new_day);

        // 3. Вычисляем прогноз: Y = X * B
        double predicted_Y = 0;
        for(int j=0; j<(int)res.B.size(); ++j)
            predicted_Y += aug_new_day[0][j] * res.B[j];

        // 4. Вычисляем матрицу (XT*X)^(-1) для стандартных ошибок
        Mat XT = transpose(augment(X_window));
        Mat G = inverse(matMul(XT, augment(X_window)));

        // 5. Вычисляем остаточную дисперсию
        int N = X_window.size();
        int k = augment(X_window)[0].size();
        double Dad = 0;
        for(int j=0; j<N; ++j) {
            Dad += (Y_window[j]-res.YR[j])*(Y_window[j]-res.YR[j]);
        }
        Dad /= (N - k);

        // 6. Вычисляем стандартную ошибку прогноза
        double s = 0;
        for(int j=0; j<k; ++j)
            for(int m=0; m<k; ++m)
                s += aug_new_day[0][j] * G[j][m] * aug_new_day[0][m];
        double SE_pred = sqrt(s * Dad);

        // 7. Вычисляем доверительный интервал прогноза
        double t_val = tValue95(N - k);
        double Y_pred_low = predicted_Y - t_val * SE_pred;
        double Y_pred_high = predicted_Y + t_val * SE_pred;

        // 8. Сохраняем результаты
        predictions.push_back(predicted_Y);
        predictions_low.push_back(Y_pred_low);
        predictions_high.push_back(Y_pred_high);
        actuals.push_back(additional_Y[i]);
        days.push_back(day_num);

        cout << "День " << day_num << ": Температура = " << additional_X[i][1]
             << ", Фактическое Y = " << additional_Y[i]
             << ", Прогнозное Y = " << predicted_Y << endl;

        // 9. Обновляем скользящее окно: удаляем самый старый день, добавляем новый
        X_window.erase(X_window.begin());
        X_window.push_back(additional_X[i]);
        Y_window.erase(Y_window.begin());
        Y_window.push_back(additional_Y[i]);
    }
    return {predictions, predictions_low, predictions_high, actuals, days};
}
Главная функция с тестовыми данными
int main() {
    // Исходные данные: 20 дней с двумя признаками [номер_дня, температура]
    Mat raw_X_initial = {
        {1, 21.5}, {2, 21.2}, {3, 22.1}, {4, 25.1}, {5, 26.4}, 
        {6, 22.6}, {7, 17.7}, {8, 18.5}, {9, 21.2}, {10, 20.3},
        {11, 17}, {12, 19.2}, {13, 19.4}, {14, 21.9}, {15, 25.5}, 
        {16, 26.3}, {17, 26.3}, {18, 24.7}, {19, 21.4}, {20, 21.04}
    };
    
    // Фактическое электропотребление для этих дней
    Vec Y_initial = {
        2357.85, 2669.7, 2669.7, 2998.05, 3512.85, 3542.55, 3248.85, 3341.25, 
        3453.45, 3598.65, 3413.85, 4271.85, 4393.95, 3686.1, 3682.8, 3550.8, 
        4719, 3979.35, 4131.6, 4141.5
    };
    // Новые дни для прогнозирования
    Mat additional_X = {
        {21, 21.3}, {22, 23}, {23, 23.45}, {24, 23.8}, {25, 21.42}, {26, 23.09}
    };
    Vec additional_Y = {4027.65, 3986.4, 3963.3, 4026, 3936.9, 3996.3};

    cout << "Регрессия на первых 20 днях:" << endl;
    RegressionResult initial_result = run_regression(raw_X_initial, Y_initial);

    cout << "\nПрогнозирование с использованием скользящего окна:" << endl;
    Prediction pred = rolling_window_prediction(raw_X_initial, Y_initial, 
                                               additional_X, additional_Y);
    return 0;
}


