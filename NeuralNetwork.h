#include "utility.h"

typedef vector<class NeuralLayer> NeuralNetwork;

enum LayerType
{
	Input,
	Hidden,
	Output
};

struct Synapse
{
	class Neuron *destination;
	double weight;
	double gradient;
};

class Neuron
{

private:
	
	double delta;
	double bias;
	double activation;
	double gradient;

	vector<Synapse> forwardConnections;
	class NeuralLayer* owner;

public:
	Neuron(class NeuralLayer* newOwner, class NeuralLayer* prevLayer);
	void AddConnection(Neuron *neuron);

	void BeginPulse(double value);
	void Pulse();

	void Activate(double inputActivation);
	double GetValue() { return activation; }

	void ComputeDelta(double expected);
	void ComputeDelta();
	void Learn(double average, double learningRate);
};


class NeuralLayer : public vector<Neuron*>
{
public:
	LayerType type;

	NeuralLayer(LayerType layerType, int numberOfNeurons, NeuralLayer* prevLayer = nullptr);

	void Pulse(double* data);
	void BackPropagate(double* expected);
	void GradientDescend(double average, double learningRate);

	void CheckResult(double* expected);
};



NeuralNetwork BuildNetwork(double hiddenLayersNum, double inputLayerDensity, double hiddenLayerDensity, double outputLayerDensity);