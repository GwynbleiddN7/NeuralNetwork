#include "NeuralNetwork.h"

Neuron::Neuron(NeuralLayer* newOwner, NeuralLayer* prevLayer)
{
	activation = 0;
	gradient = 0;
	delta = 0;
	bias = getRandomDouble();

	owner = newOwner;
	
	if(prevLayer != nullptr)
	{
		for(int i=0; i < prevLayer->size(); i++)
		{
			prevLayer->at(i)->AddConnection(this);
		}
	}
}

void Neuron::AddConnection(Neuron *neuron)
{
	Synapse newLink;
	newLink.destination = neuron;
	newLink.weight = getRandomDouble();
	newLink.gradient = 0;

	forwardConnections.push_back(newLink);
}



void Neuron::BeginPulse(double value)
{
	activation = value;
	Pulse();
}

void Neuron::Pulse()
{
	if (owner->type != LayerType::Input) activation = sigmoid(activation + bias); // apply function to propagated value from previous layer

	for (Synapse synapse : forwardConnections)
	{
		synapse.destination->Activate(activation * synapse.weight);
	}
}


void Neuron::Activate(double inputActivation)
{
	activation += inputActivation;
}


void Neuron::ComputeDelta(double expected)
{
	delta = 2 * (expected - activation) * sigmoid_derivative(activation);
	gradient += delta;
	activation = 0;
}

void Neuron::ComputeDelta()
{
	delta = 0;
	for (Synapse synapse : forwardConnections)
	{
		delta += synapse.weight * synapse.destination->delta * sigmoid_derivative(activation);
		synapse.gradient += synapse.destination->delta * activation;
	}
	gradient += delta;
	activation = 0;
}

void Neuron::Learn(double average, double learningRate)
{
	gradient /= average;
	bias -= gradient * learningRate;
	gradient = 0;

	for (Synapse synapse : forwardConnections)
	{
		synapse.gradient /= average;
		synapse.weight -= synapse.gradient * learningRate;
		synapse.gradient = 0;
	}
}


NeuralLayer::NeuralLayer(LayerType layerType, int numberOfNeurons, NeuralLayer* prevLayer)
{
	type = layerType;
	for (int i = 0; i < numberOfNeurons; i++)
	{
		Neuron *neuron = new Neuron(this, prevLayer);
		push_back(neuron);
	}
}

void NeuralLayer::Pulse(double* data)
{
	for(int n=0; n<this->size(); n++)
	{
		if (this->type == LayerType::Input) this->at(n)->BeginPulse(data[n]);
		else this->at(n)->Pulse();
		
		//cout <<  this->at(n)->GetValue() << "y " <<endl;
	 
	}
}

void NeuralLayer::BackPropagate(double* expected)
{
	for(int n=0; n<this->size(); n++)
	{
		if (this->type == LayerType::Output) this->at(n)->ComputeDelta(expected[n]);
		else this->at(n)->ComputeDelta();
	}
}

void NeuralLayer::GradientDescend(double average, double learningRate)
{
	for(int n=0; n<this->size(); n++)
	{
		this->at(n)->Learn(average, learningRate);
	}
}

void NeuralLayer::CheckResult(double* expected)
{
	if (this->type != LayerType::Output) return;
	for(int n=0; n<this->size(); n++)
	{
		cout << this->at(n)->GetValue() << " : " << expected[n] << endl;
	}
}


NeuralNetwork BuildNetwork(double hiddenLayersNum, double inputLayerDensity, double hiddenLayerDensity, double outputLayerDensity)
{
	NeuralNetwork neuralNet;
	
	NeuralLayer inputLayer = NeuralLayer(LayerType::Input, inputLayerDensity);
	neuralNet.push_back(inputLayer);

	for (int i = 0; i < hiddenLayersNum; i++)
	{
		NeuralLayer hiddenLayer = NeuralLayer(LayerType::Hidden, hiddenLayerDensity, &neuralNet[i]);
		neuralNet.push_back(hiddenLayer);
	}

    NeuralLayer outputLayer = NeuralLayer(LayerType::Output, outputLayerDensity, &neuralNet[hiddenLayersNum]);
	neuralNet.push_back(outputLayer);

	return neuralNet;
}