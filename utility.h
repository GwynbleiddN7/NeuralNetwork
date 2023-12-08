#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef unsigned char uchar;

double sigmoid(double x);
double sigmoid_derivative(double x);
double ReLU(double x);

double getRandomDouble();

system_clock::time_point getTime();
void getDuration(system_clock::time_point start);