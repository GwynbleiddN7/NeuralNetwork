#include "NeuralNetwork.h"

int main(int argc, char const *argv[])
{
    system_clock::time_point start;

	// Create Network
    cout << "Building network..." << endl;
    start = getTime();

	const int iterations = atoi(argv[1]);
	const int hiddenLayersNum = atoi(argv[2]);
    const int hiddenLayerDensity = atoi(argv[3]);
	const int inputLayerDensity = 2;
	const int outputLayerDensity = 2;

	const double learningRate = 0.05;
	const int gradientDescentRate = 1;

    NeuralNetwork neuralNet = BuildNetwork(hiddenLayersNum, inputLayerDensity, hiddenLayerDensity, outputLayerDensity);
    if(neuralNet.size() < 3)
    {
        cout << "Neural network must contain at least 3 layers" << endl;
        return -1;
    }
    getDuration(start);


	double input[48][2] = 
	{
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	double labels[48][2] = 
	{
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 1 },
		{ 1, 0 }
	};


    // Propagate
    cout << "Learning..." << endl;
    start = getTime();
	for (int i = 0, j = 0; i < iterations; i++, j++)
	{
		//1) Pulse
		for (auto layer : neuralNet)
		{
			layer.Pulse(input[i]);
		}

		//2) Back Propagate
		for (int k = neuralNet.size()-1; k >= 0; k--)
		{
			neuralNet[k].BackPropagate(labels[i]);
		}

		//3) Gradient Descent
		if(j == gradientDescentRate)
		{
			for (int k = neuralNet.size()-1; k >= 0; k--)
			{
				neuralNet[k].GradientDescend(gradientDescentRate, learningRate);
			}
			j = 0;
		}
		continue;
		if(i % (iterations/10) == 0 && i>0)
		{
			cout << (i/(iterations/10))*10 << "%\n";
		}
	}
    getDuration(start);




	// Testing
	cout << "Testing..." << endl;
    start = getTime();

	for (auto layer : neuralNet)
	{
		layer.Pulse(input[0]);
	}

	neuralNet[neuralNet.size()-1].CheckResult(labels[0]);
	cout << endl << endl;
	
	getDuration(start);
	


	return 0;
}