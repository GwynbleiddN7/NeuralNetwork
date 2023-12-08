#include "NeuralNetwork.h"
#include "mnist.h"

int main(int argc, char const *argv[])
{
    system_clock::time_point start;

	// Create Network
    cout << "Building network..." << endl;
    start = getTime();

	const int iterations = atoi(argv[1]);
	const int hiddenLayersNum = atoi(argv[2]);
    const int hiddenLayerDensity = atoi(argv[3]);
	const int inputLayerDensity = 784;
	const int outputLayerDensity = 10;

	const double learningRate = 0.05;
	const int gradientDescentRate = 50;

    NeuralNetwork neuralNet = BuildNetwork(hiddenLayersNum, inputLayerDensity, hiddenLayerDensity, outputLayerDensity);
    if(neuralNet.size() < 3)
    {
        cout << "Neural network must contain at least 3 layers" << endl;
        return -1;
    }
    getDuration(start);

	// Load
    cout << "Loading data..." << endl;
    start = getTime();
	int img_num, img_size;
	double** images = getTrainingImages(img_num, img_size);
	double** labels = getTrainingLabels(img_num);

	for (int i = 0; i < iterations; i++)
	{
		for(int j=0; j<10; j++)
		{
			cout << labels[i][j] << endl;
		}
	}

	if(neuralNet[0].size() != img_size)
	{
		cout << "Input values count must be equal to input layer count" << endl;
		return -1;
	}
    getDuration(start);


    // Propagate
    cout << "Learning..." << endl;
    start = getTime();
	for (int i = 0, j = 0; i < iterations; i++, j++)
	{
		//1) Pulse
		for (auto layer : neuralNet)
		{
			layer.Pulse(images[i]);
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

		if(i % (iterations/10) == 0)
		{
			cout << (i/(iterations/10))*10 << "%\n";
		}
	}
    getDuration(start);


	int t_img_num, t_img_size;
	double** test_images = getTrainingImages(t_img_num, t_img_size);
	double** test_labels = getTrainingLabels(t_img_num);

	// Testing
	cout << "Testing..." << endl;
    start = getTime();

	for (auto layer : neuralNet)
	{
		layer.Pulse(images[0]);
	}

	neuralNet[neuralNet.size()-1].CheckResult(labels[0]);
	cout << endl << endl;
	
	getDuration(start);
	


	return 0;
}