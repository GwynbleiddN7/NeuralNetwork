#include "utility.h"

double sigmoid(double x)
{
	return 1.0 / (1.0 + (exp(-x)));
}

double sigmoid_derivative(double x)
{
	return x * (1.0 - x);
}

double ReLU(double x)
{
	return max(0.0, x);
}

double getRandomDouble()
{
	srand(time(0));
	return ((double)rand() / (double)RAND_MAX)/5;
}

system_clock::time_point getTime()
{
	return high_resolution_clock::now();
}

void getDuration(system_clock::time_point start)
{
	cout.precision(3);

	auto duration = duration_cast<nanoseconds>(getTime() - start).count();

	double res = duration / pow(10, 3);
	if(res > pow(10, 6))
	{
		res = res / pow(10, 6);
		cout << "Last execution time: " << res << "s\n" << endl;
	}
	else if(res > pow(10, 3))
	{
		res = res / pow(10, 3);
		cout << "Last execution time: " << res << "ms\n" << endl;
	}
	else cout << "Last execution time: " << res << "us\n" << endl;
}