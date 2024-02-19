using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Cortana
{
    public class NeuralNetwork
    {
        static void Run()
        {
            // NeuralNet.TrainNet = false;

            var bigdata = FileReaderMNIST.LoadImagesAndLables("./data/train-labels-idx1-ubyte.gz", "./data/train-images-idx3-ubyte.gz");
            var data = bigdata.ToArray();

            double[,] InputValues = new double[data.Length, 784];
            double[,] OutputValues = new double[data.Length, 10];

            for (int i = 0; i < InputValues.GetLength(0); i++)
            {
                var arraydata = data[i].Image.Cast<byte>().ToArray();
                for (int j = 0; j < arraydata.Length; j++)
                {
                    InputValues[i, j] = arraydata[j] / 255.0;
                }
                int number = data[i].Label;
                for (int j = 0; j < 10; j++) OutputValues[i, j] = 0;
                OutputValues[i, number] = 1;
            }

            //Logic Gate Test
            //double[,] InputValues = new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            //double[,] OutputValues = new double[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 } };

            NeuralNet neuralNet = new NeuralNet(InputNeuronsNum: InputValues.GetLength(1), OutputNeuronsNum: OutputValues.GetLength(1), HiddenNeuronsNum: 20, HiddenLayersNum: 2);

            string result = neuralNet.Train(5000000, InputValues, OutputValues);
            Console.WriteLine(result);

            Console.WriteLine("Known images:");
            double[] TestImages = new double[784];
            for (int i = 0; i < 40; i++)
            {
                var ImagesArray = data[i].Image.Cast<byte>().ToArray();
                var NamesArray = data[i].Label;
                for (int j = 0; j < ImagesArray.Length; j++)
                {
                    TestImages[j] = ImagesArray[j] / 255.0;
                }
                Console.WriteLine($"Number should be {NamesArray}");
                List<double> res = neuralNet.SendInput(TestImages);
                Console.WriteLine($"Number with highest accuracy is {res.IndexOf(res.Max())}, with {res.Max()}% accuracy");
                Console.Write("Accuracy List: ");
                res.ForEach(x => Console.Write($"{res.IndexOf(x)}=>{Math.Round(x, 2)}%   "));
                Console.Write("\n\n");
            }

            var hiddendata = FileReaderMNIST.LoadImagesAndLables("./data/t10k-labels-idx1-ubyte.gz", "./data/t10k-images-idx3-ubyte.gz");
            var datahidden = hiddendata.Take(40).ToArray();

            Console.WriteLine("Unknown images:");
            double[] TestImages2 = new double[784];
            for (int i = 0; i < 40; i++)
            {
                var ImagesArray = datahidden[i].Image.Cast<byte>().ToArray();
                var NamesArray = datahidden[i].Label;
                for (int j = 0; j < ImagesArray.Length; j++)
                {
                    TestImages2[j] = ImagesArray[j] / 255.0;
                }
                Console.WriteLine($"Number should be {NamesArray}");
                List<double> res = neuralNet.SendInput(TestImages2);
                Console.WriteLine($"Number with highest accuracy is {res.IndexOf(res.Max())}, with {res.Max()}% accuracy");
                Console.Write("Accuracy List: ");
                res.ForEach(x => Console.Write($"{res.IndexOf(x)}=>{Math.Round(x, 2)}%   "));
                Console.Write("\n\n");
            }
            //double[] InputValue = new double[] { 0, 0 };
            //Console.WriteLine("Logic Gate Test");
            //neuralNet.SendInput(InputValue);
        }
    }

    public class Neuron
    {
        static private double LearningRate = 0.05;
        public Dictionary<Neuron, double> Weights { get; set; } = new Dictionary<Neuron, double>();
        public double Output { get; set; } = 0;
        public double Bias { get; set; } = 0;
        public double Error { get; set; } = 0;

        //HiddenLayer Initializer + OutputLayer Initializer
        public Neuron(NeuronLayer PreviousLayer)
        {
            foreach (Neuron PreviousNeuron in PreviousLayer)
            {
                Weights.Add(PreviousNeuron, new Random().NextDouble() / 5);
                Bias = new Random().NextDouble() / 5;
            }
        }

        //Input Layer Initializer
        public Neuron() { }

        public void Pulse(ref NeuronLayer NextLayer)
        {
            foreach (var NextNeuron in NextLayer)
            {
                NextNeuron.Output += NextNeuron.Weights[this] * Output;
            }
        }

        public void CalculateError(double ExpectedOutput)
        {
            //if(ExpectedOutput == 1) Console.WriteLine($"{100 * Output / ExpectedOutput}%");

            double error = ExpectedOutput - Output;
            Error = error * dSigma(Output);
        }

        public void PropagateError(NeuronLayer NextLayer)
        {
            double error = 0;
            foreach (var NextNeuron in NextLayer)
            {
                error += NextNeuron.Weights[this] * NextNeuron.Error * dSigma(Output);
            }
            Error = error;
        }

        public void ApplyLearning(NeuronLayer NextLayer)
        {
            foreach (Neuron NextNeuron in NextLayer)
            {
                NextNeuron.Weights[this] += Output * NextNeuron.Error * LearningRate;
                NextNeuron.Bias += NextNeuron.Error * LearningRate;
            }
        }

        public void AdjustOutput()
        {
            Output = Sigma(Output + Bias);
        }

        static double Sigma(double input)
        {
            return (1.0F / (1.0F + Math.Exp(-input)));
        }

        static double dSigma(double output)
        {
            return output * (1.0F - output);
        }
    }

    public enum ELayerType
    {
        Input,
        Hidden,
        Output
    }

    public class NeuronLayer : List<Neuron>
    {
        public ELayerType LayerType { get; set; }

        public NeuronLayer(ELayerType Type)
        {
            LayerType = Type;
        }

        public void AddNeurons(int Num)
        {
            for (int i = 0; i < Num; i++) Add(new Neuron());
        }
        public void AddNeurons(int Num, NeuronLayer SynapseBinder)
        {
            for (int i = 0; i < Num; i++) Add(new Neuron(PreviousLayer: SynapseBinder));
        }

        public void Pulse(NeuronLayer TargetLayer)
        {
            foreach (Neuron CurrentNeuron in this)
            {
                CurrentNeuron.Pulse(ref TargetLayer);
            }
        }

        public void SigmaOutput()
        {
            foreach (Neuron CurrentNeuron in this)
            {
                CurrentNeuron.AdjustOutput();
            }
        }

        public bool CalculateError(double[,] ExpectedResult, int Index)
        {
            for (int i = 0; i < ExpectedResult.GetLength(1); i++)
            {
                try
                {
                    this[i].CalculateError(ExpectedResult[Index, i]);
                }
                catch
                {
                    return false;
                }
            }
            return true;
        }

        public void BackPropagateError(NeuronLayer SourceLayer)
        {
            foreach (Neuron CurrentNeuron in this)
            {
                CurrentNeuron.PropagateError(SourceLayer);
            }
        }

        public void BackPropagateCorrection(NeuronLayer SourceLayer)
        {
            foreach (Neuron CurrentNeuron in this)
            {
                CurrentNeuron.ApplyLearning(SourceLayer);
            }
        }
    }

    public class NeuralNet : List<NeuronLayer>
    {
        public static bool TrainNet = true;
        public NeuronLayer InputLayer
        {
            get { return this.First(); }
        }
        public List<NeuronLayer> HiddenLayers
        {
            get { return this.GetRange(1, Count - 2); }
        }
        public NeuronLayer OutputLayer
        {
            get { return this.Last(); }
        }


        public NeuralNet(int InputNeuronsNum = 2, int OutputNeuronsNum = 1, int HiddenNeuronsNum = 4, int HiddenLayersNum = 2)
        {
            if (InputNeuronsNum <= 0) InputNeuronsNum = 1;
            if (OutputNeuronsNum <= 0) OutputNeuronsNum = 1;
            if (HiddenNeuronsNum <= 0) HiddenNeuronsNum = 1;
            if (HiddenLayersNum <= 0) HiddenLayersNum = 1;

            //Adding Input Layer
            NeuronLayer inputLayer = new NeuronLayer(ELayerType.Input);
            inputLayer.AddNeurons(InputNeuronsNum);
            this.Add(inputLayer);

            //Adding Hidden Layers
            for (int i = 1; i <= HiddenLayersNum; i++)
            {
                this.Add(new NeuronLayer(ELayerType.Hidden));
                this[i].AddNeurons(HiddenNeuronsNum, SynapseBinder: this[i - 1]);
            }

            //Adding Output Layer
            NeuronLayer outputLayer = new NeuronLayer(ELayerType.Output);
            outputLayer.AddNeurons(OutputNeuronsNum, SynapseBinder: this.Last());
            this.Add(outputLayer);

            Console.WriteLine("Neural Net Built");
        }

        //Propagate Signal
        private void Pulse()
        {
            for (int i = 0; i < this.Count - 1; i++)
            {
                this[i].Pulse(TargetLayer: this[i + 1]);
                this[i + 1].SigmaOutput();
            }
        }

        //Train with Input Data
        public string Train(int NumberOfSessions, double[,] Inputs, double[,] ExpectedOutputs)
        {
            Console.WriteLine("Neural Net Training...");
            int ActualTrainSessions = 0;
            for (int i = 0; i < NumberOfSessions; i++)
            {
                ResetOutputs();
                int index = new Random().Next(Inputs.GetLength(0));
                try
                {
                    for (int j = 0; j < Inputs.GetLength(1); j++)
                    {
                        InputLayer[j].Output = Inputs[index, j];
                    }
                }
                catch
                {
                    return "[NEURALNET ERROR] Numero di InputNeurons diverso dal numero di Inputs";
                }
                Pulse();
                if (!Learn(ExpectedOutputs, index)) return "[NEURALNET ERROR] Numero di OutputNeurons diverso dal numero di Outputs";
                ActualTrainSessions++;
                if (ActualTrainSessions % 1000 == 0)
                {
                    Console.WriteLine($"{Math.Round(100.0 * (double)ActualTrainSessions / (double)NumberOfSessions, 5)}%");
                    var bigdata = FileReaderMNIST.LoadImagesAndLables("./data/train-labels-idx1-ubyte.gz", "./data/train-images-idx3-ubyte.gz");
                    var data = bigdata.Take(1).ToArray();

                    double[] TestImages = new double[784];
                    var ImagesArray = data[0].Image.Cast<byte>().ToArray();
                    var NamesArray = data[0].Label;
                    for (int j = 0; j < ImagesArray.Length; j++)
                    {
                        TestImages[j] = ImagesArray[j] / 255.0;
                    }
                    Console.WriteLine($"Number should be {NamesArray}");
                    List<double> res = SendInput(TestImages);
                    Console.WriteLine($"Number with highest accuracy is {res.IndexOf(res.Max())}, with {Math.Round(res.Max(), 2)}% accuracy");
                    Console.Write("Accuracy List: ");
                    res.ForEach(x => Console.Write($"{res.IndexOf(x)}=>{Math.Round(x, 2)}%   "));
                    Console.Write("\n\n");


                }
                if (!TrainNet) break;
            }
            return $"Training terminato con {ActualTrainSessions} sessioni";
        }

        public List<double> SendInput(double[] input)
        {
            ResetOutputs();
            for (int i = 0; i < input.Length; i++) InputLayer[i].Output = input[i];
            Pulse();
            List<double> results = new List<double>();
            OutputLayer.ForEach(neuron => results.Add(Math.Round(neuron.Output, 3) * 100));

            return results;
        }

        private bool Learn(double[,] ExpectedResult, int Index)
        {
            //Adjust Error on Output Layer
            bool result = OutputLayer.CalculateError(ExpectedResult, Index);

            //Loop from the last Hidden Layer to the first Hidden Layer
            for (int i = this.Count - 2; i > 0; i--)
            {
                this[i].BackPropagateError(SourceLayer: this[i + 1]);
            }

            //Loop from the last Hidden Layer to the Input Layer
            for (int i = this.Count - 2; i >= 0; i--)
            {
                this[i].BackPropagateCorrection(SourceLayer: this[i + 1]);
            }
            return result;
        }

        private void ResetOutputs()
        {
            this.ForEach(Layer => Layer.ForEach(neuron => neuron.Output = 0));
        }
    }

}
