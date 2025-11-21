using System;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static Matrix<double> Augment(Matrix<double> X)
    {
        int N = X.RowCount;
        var augmentedX = Matrix<double>.Build.Dense(N, 5);

        for (int i = 0; i < N; i++)
        {
            augmentedX[i, 0] = X[i, 0]; // X0
            augmentedX[i, 1] = X[i, 1]; // X1
            augmentedX[i, 2] = Math.Pow(X[i, 1], 2); // X1^2
            augmentedX[i, 3] = X[i, 2]; // X3
            augmentedX[i, 4] = X[i, 1] * X[i, 2]; // X1 * X3
        }
        return augmentedX;
    }

    static double Predict(Matrix<double> X, Matrix<double> Y, bool printData = false)
    {
        var XT = X.Transpose();

        // Коэффициенты регрессии
        var B = (XT * X).Inverse() * XT * Y;

        if (printData)
        {
            Console.WriteLine("Regression Coefficients B:");
            for (int i = 0; i < B.RowCount; i++)
                Console.WriteLine($"B{i}: {B[i, 0]}");
        }

        // Расчетные значения зависимой переменной
        var YR = X * B;

        if (printData)
        {
            Console.WriteLine("\nPredicted Values YR:");
            for (int i = 0; i < YR.RowCount; i++)
                Console.WriteLine($"YR{i}: {YR[i, 0]}");
        }

        int N = X.RowCount;
        int k = X.ColumnCount;
        // Дисперсия адекватности
        var Dad = ((Y - YR).PointwisePower(2).RowSums().Sum()) / (N - k);
        // Средняя арифметическая зависимой переменной
        var YSR = Y.ColumnSums().Sum() / N;
        // Дисперсия зависимой переменной
        var DY = (Y - YSR).PointwisePower(2).RowSums().Sum() / (N - 1);
        // Расчетное значение F-статистики
        var FR = DY / Dad;

        if (printData)
        {
            Console.WriteLine($"\nAdequacy Variance (Dad): {Dad}");
            Console.WriteLine($"YSR: {YSR}");
            Console.WriteLine($"Dependent Variable Variance (DY): {DY}");
            Console.WriteLine($"F-Statistic (FR): {FR}");
        }

        // Матрица обратная матрице нормальных уравнений
        var G = (XT * X).Inverse();
        double t = 2.131; // Табличное значение критерия Стьюдента
        // Доверительные интервалы коэффициентов регрессии
        var deltasB = G.Diagonal().Multiply(Dad).PointwiseSqrt().Multiply(t);

        if (printData)
        {
            Console.WriteLine("\nConfidence Intervals for Regression Coefficients:");
            for (int i = 0; i < B.RowCount; i++)
            {
                double delta = deltasB[i];
                Console.WriteLine($"B{i}: {B[i, 0] - delta} <= {B[i, 0]} <= {B[i, 0] + delta} | delta = {delta}");
            }
        }

        // Ошибка прогноза
        var D = X * (XT * X).Inverse() * XT;
        // Доверительный интервал коридора ошибок
        var deltasS = D.Diagonal().Add(1.0).Multiply(Dad).PointwiseSqrt().Multiply(t);

        double tau = 21;
        double A = 21.3;

        // Вектор независимых переменных в прогнозной точке
        var predictionPoint = Vector<double>.Build.DenseOfArray(new double[] { 1, tau, tau * tau, A, tau * A });
        // Математическая модель
        var YP = predictionPoint * B;
        // Значение зависимой переменной в прогнозной точке
        if (printData)
            Console.WriteLine($"\nPredicted Value (YP): {YP[0]}");

        double S20 = deltasS.Last();

        if (printData)
        {
            Console.WriteLine("\nConfidence Intervals for Observation S20:");
            Console.WriteLine($"S20: {S20}");
        }

        //Коридор ошибок в пронозной точке
        if (printData)
        {
            Console.WriteLine("\nError Corridor:");
            Console.WriteLine($"Ymax: {YP[0] + S20}");
            Console.WriteLine($"Ymin: {YP[0] - S20}");
        }

        return YP[0];
    }

    static (Matrix<double>, Matrix<double>) MoveMatricies(Matrix<double> oldX, Matrix<double> oldY, Vector<double> newXRow, double newY)
    {
        int rowCount = oldX.RowCount;
        int colCount = oldX.ColumnCount;

        if (newXRow.Count != colCount)
        {
            throw new ArgumentException("New row must have the same number of columns as the matrix.");
        }

        var resultX = Matrix<double>.Build.Dense(rowCount, colCount);
        var resultY = Matrix<double>.Build.Dense(rowCount, 1);

        for (int i = 1; i < rowCount; i++)
        {
            resultX.SetRow(i - 1, oldX.Row(i));
            resultY.SetRow(i - 1, oldY.Row(i));
        }

        resultX.SetRow(rowCount - 1, newXRow);
        resultY.SetRow(rowCount - 1, [newY]);

        return (resultX, resultY);
    }


    static void Main()
    {
        // Исходные переменные данные X
        var rawX = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            { 1, 1, 21.5 }, { 1, 2, 21.2 }, { 1, 3, 22.1 }, { 1, 4, 25.1 }, { 1, 5, 26.4 },
            { 1, 6, 22.6 }, { 1, 7, 17.7 }, { 1, 8, 18.5 }, { 1, 9, 21.2 }, { 1, 10, 20.3 },
            { 1, 11, 17 }, { 1, 12, 19.2 }, { 1, 13, 19.4 }, { 1, 14, 21.9 }, { 1, 15, 25.5 },
            { 1, 16, 26.3 }, { 1, 17, 26.3 }, { 1, 18, 24.7 }, { 1, 19, 21.4 }, { 1, 20, 21.04 }
        });
        // Исходные данные Y
        var Y = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            { 2357.85 }, { 2669.7 }, { 2669.7 }, { 2998.05 }, { 3512.85 }, { 3542.55 },
            { 3248.85 }, { 3341.25 }, { 3453.45 }, { 3598.65 }, { 3413.85 }, { 4271.85 },
            { 4393.95 }, { 3686.1 }, { 3682.8 }, { 3550.8 }, { 4719 }, { 3979.35 },
            { 4131.6 }, { 4141.5 }
        });

        // Дополненные данные Х
        var X = Augment(rawX);

        int day = X.RowCount;

        bool stopped = false;
        char userInput;

        Matrix<double> predictions = Matrix<double>.Build.Dense(100, 6);
        int predictionsCount = 0;

        do
        {
            day += 1;
            var predY = Predict(X, Y);
            Console.WriteLine($"Prediction for day {day}: {predY}");

            Console.WriteLine(new String('-', 30));
            do
            {
                Console.WriteLine("Continue calculations for the next day? (Y/N)");
                userInput = Console.ReadLine()[0];
            } while (userInput != 'N' && userInput != 'Y' &&
                     userInput != 'y' && userInput != 'n' &&
                     userInput != '1' && userInput != '0');
            if (userInput is 'N' or 'n' or '0')
                stopped = true;
            else
            {
                Console.WriteLine($"Input real Y for day {day}:");
                double realY = double.Parse(Console.ReadLine().Replace('.', ','));
                Console.WriteLine($"Input temperature for day {day}:");
                double realTemp = double.Parse(Console.ReadLine().Replace('.', ','));

                double absoluteError = realY - predY;
                Console.WriteLine($"Absolute error for day {day} is {absoluteError}");
                double relativeError = Math.Abs(absoluteError) / realY * 100;
                Console.WriteLine($"Relative error for day {day} is {relativeError}%");

                Vector<double> newXRow = Vector<double>.Build.DenseOfArray([1, day, day * day, realTemp, day * realTemp]);
                (X, Y) = MoveMatricies(X, Y, newXRow, realY);

                if (predictionsCount < predictions.RowCount)
                {
                    Vector<double> predictionsStatsRow =
                        Vector<double>.Build.DenseOfArray([day, realTemp, realY, predY, absoluteError, relativeError]);
                    predictions.SetRow(predictionsCount, predictionsStatsRow);
                    predictionsCount++;
                }
            }
        } while (stopped == false);

        Console.WriteLine(predictions.ToString(predictionsCount + 10, 6));
    }
}
