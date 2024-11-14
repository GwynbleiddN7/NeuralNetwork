namespace NeuralNetwork;

public class Synapse(Neuron destination, double weight)
{
    public Neuron LinkedNeuron { get; } = destination;
    public double Weight { get; set; } = weight;
    public double Error { get; set; } = 0;
}

public class Neuron
{
    private List<Synapse> Synapses { get; } = [];
    private double Bias { get; set; }
    public double Value { get; set; }
    private double Error { get; set; }

    //Input Layer Initializer
    public Neuron() { }
    
    //HiddenLayer Initializer + OutputLayer Initializer
    public Neuron(NeuronLayer previousLayer)
    {
        foreach (Neuron newLinkedNeuron in previousLayer) {
            newLinkedNeuron.AddSynapse(this);
        }
    }

    private void AddSynapse(Neuron neuron)
    {
        var synapse = new Synapse(neuron, new Random().NextDouble());
        Synapses.Add(synapse);
    }

    public void SetValue(double newValue) => Value = newValue;
    public void AddValue(double addValue) => Value += addValue;
    
    public void Push(double initialValue)
    {
        Value = initialValue;
        Pulse();
    }
    
    public void Pulse()
    {
        foreach(Synapse neuralLink in Synapses)
        { 
            neuralLink.LinkedNeuron.AddValue(Value * neuralLink.Weight);
        }
    }
}

    public enum ELayerType
    {
        Input,
        Hidden,
        Output
    }

    public class InputLayer() : NeuronLayer(ELayerType.Input)
    {
        public InputLayer(int num) : this()
        {
            for (var i = 0; i < num; i++) Add(new Neuron());
        }
    }

    public class HiddenLayer() : NeuronLayer(ELayerType.Hidden)
    {
        public HiddenLayer(int num, NeuronLayer previousLayer) : this()
        {
            for (var i = 0; i < num; i++) Add(new Neuron(previousLayer));
        }

        public void ApplyActivationFunction()
        {
            //Activation function (ReLu)
        }
    }

    public class OutputLayer() : NeuronLayer(ELayerType.Output)
    {
        public OutputLayer(int num, NeuronLayer previousLayer) : this()
        {
            for (var i = 0; i < num; i++) Add(new Neuron(previousLayer));
        }

        public void CalcuateOutput()
        {
            //Activation function (softmax)
            //Cross Entropy
        }
    }

    public abstract class NeuronLayer(ELayerType type) : List<Neuron>
    {
        public ELayerType LayerType { get; set; } = type;

        public void Propagate() => ForEach(neuron => neuron.Pulse());

        public bool CalculateError(double[,] expectedResult, int index)
        {
            for (var i = 0; i < expectedResult.GetLength(1); i++)
            {
                try
                {
                    this[i].CalculateError(expectedResult[index, i]);
                }
                catch
                {
                    return false;
                }
            }
            return true;
        }
    }

    public class NeuralNetwork : List<NeuronLayer>
    {
        private readonly InputLayer _inputLayer;
        private readonly OutputLayer _outputLayer;


        public NeuralNetwork(int inputNeuronsNum = 2, int outputNeuronsNum = 1, int hiddenNeuronsNum = 4, int hiddenLayersNum = 2)
        {
            //Adding Input Layer
            _inputLayer = new InputLayer(inputNeuronsNum);
            Add(_inputLayer);

            //Adding Hidden Layers
            for (var i = 1; i <= hiddenLayersNum; i++)
            {
                Add(new HiddenLayer(hiddenNeuronsNum, this[i - 1]));
            }

            //Adding Output Layer
            _outputLayer = new OutputLayer(outputNeuronsNum, this.Last());
            Add(_outputLayer);

            Console.WriteLine("Neural Network built");
        }

        //Propagate Signal
        private void Propagate()
        {
            foreach (NeuronLayer layer in this)
            {
                switch (layer)
                {
                    case HiddenLayer hiddenLayer:
                        hiddenLayer.ApplyActivationFunction();
                        break;
                    case OutputLayer outputLayer:
                        outputLayer.CalcuateOutput();
                        continue;
                }
                layer.Propagate();
            }
        }

        //Train with Input Data
        public string Train(int epochs, double[,] inputs, double[,] expectedOutputs)
        {
            Console.WriteLine("Neural Net Training...");
            var currentEpoch = 0;

            for (var i = 0; i < epochs; i++)
            {
                ResetOutputs();
                int index = new Random().Next(inputs.GetLength(0));
                try
                {
                    for (var j = 0; j < inputs.GetLength(1); j++)
                    {
                        _inputLayer[j].Value = inputs[index, j];
                    }
                }
                catch
                {
                    return "[NEURALNET ERROR] Numero di InputNeurons diverso dal numero di Inputs";
                }
                Propagate();
                if (!Learn(expectedOutputs, index)) return "[NEURALNET ERROR] Numero di OutputNeurons diverso dal numero di Outputs";
                currentEpoch++;
                if (currentEpoch % 1000 != 0) continue;
                
                Console.WriteLine($"{Math.Round(100.0 * currentEpoch / epochs, 5)}%");

                var testImages = new double[784];
                byte[] imagesArray = DataDb.TrainData[0].Image.Cast<byte>().ToArray();
                byte namesArray = DataDb.TrainData[0].Label;
                for (var j = 0; j < imagesArray.Length; j++)
                {
                    testImages[j] = imagesArray[j] / 255.0;
                }
                Console.WriteLine($"Number should be {namesArray}");
                List<double> res = SendInput(testImages);
                Console.WriteLine($"Number with highest accuracy is {res.IndexOf(res.Max())}, with {Math.Round(res.Max(), 2)}% accuracy");
                Console.Write("Accuracy List: ");
                res.ForEach(x => Console.Write($"{res.IndexOf(x)}=>{Math.Round(x, 2)}%   "));
                Console.Write("\n\n");
                
            }
            return $"Training terminato con {currentEpoch} sessioni";
        }

        private List<double> SendInput(double[] input)
        {
            ResetOutputs();
            for (var i = 0; i < input.Length; i++) _inputLayer[i].Value = input[i];
            Propagate();
            var results = new List<double>();
            _outputLayer.ForEach(neuron => results.Add(Math.Round(neuron.Value, 3) * 100));

            return results;
        }

        private bool Learn(double[,] expectedResult, int index)
        {
            //Adjust Error on Output Layer
            bool result = _outputLayer.CalculateError(expectedResult, index);

            //Loop from the last Hidden Layer to the first Hidden Layer
            for (int i = this.Count - 2; i > 0; i--)
            {
                this[i].BackPropagateError(sourceLayer: this[i + 1]);
            }

            //Loop from the last Hidden Layer to the Input Layer
            for (int i = this.Count - 2; i >= 0; i--)
            {
                this[i].BackPropagateCorrection(sourceLayer: this[i + 1]);
            }
            return result;
        }

        private void ResetOutputs()
        {
            ForEach(layer => layer.ForEach(neuron => neuron.Output = 0));
        }
    }