using MNIST.IO;

namespace NeuralNetwork
{
    public static class Entry
    {
        private static void Main()
        {
            
            var inputValues = new double[DataDb.TrainCount, DataDb.ImageSize];
            var outputValues = new double[DataDb.TrainCount, DataDb.LabelSize];

            for (var i = 0; i < DataDb.TrainCount; i++)
            {
                byte[] arrayData = DataDb.TrainData[i].Image.Cast<byte>().ToArray();
                for (var j = 0; j < DataDb.ImageSize; j++)
                {
                    inputValues[i, j] = arrayData[j] / 255.0;
                }
                int number = DataDb.TrainData[i].Label;
                for (var j = 0; j < DataDb.LabelSize; j++) outputValues[i, j] = 0;
                outputValues[i, number] = 1;
            }

            //Logic Gate Test
            //double[,] InputValues = new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            //double[,] OutputValues = new double[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 } };

            var neuralNet = new NeuralNet(inputNeuronsNum: DataDb.ImageSize, outputNeuronsNum: DataDb.LabelSize, hiddenNeuronsNum: 32, hiddenLayersNum: 2);

            string result = neuralNet.Train(1000000, inputValues, outputValues);
            Console.WriteLine(result);

            Console.WriteLine("Known images:");
            var testImages = new double[784];
            for (var i = 0; i < 40; i++)
            {
                byte[] imagesArray = DataDb.TrainData[i].Image.Cast<byte>().ToArray();
                byte namesArray = DataDb.TrainData[i].Label;
                for (var j = 0; j < imagesArray.Length; j++)
                {
                    testImages[j] = imagesArray[j] / 255.0;
                }
                Console.WriteLine($"Number should be {namesArray}");
                List<double> res = neuralNet.SendInput(testImages);
                Console.WriteLine($"Number with highest accuracy is {res.IndexOf(res.Max())}, with {res.Max()}% accuracy");
                Console.Write("Accuracy List: ");
                res.ForEach(x => Console.Write($"{res.IndexOf(x)}=>{Math.Round(x, 2)}%   "));
                Console.Write("\n\n");
            }

            Console.WriteLine("Unknown images:");
            var testImages2 = new double[784];
            for (var i = 0; i < 40; i++)
            {
                byte[] imagesArray = DataDb.TestData[i].Image.Cast<byte>().ToArray();
                byte namesArray = DataDb.TestData[i].Label;
                for (var j = 0; j < imagesArray.Length; j++)
                {
                    testImages2[j] = imagesArray[j] / 255.0;
                }
                Console.WriteLine($"Number should be {namesArray}");
                List<double> res = neuralNet.SendInput(testImages2);
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

    

    internal struct DataDb
    {
        public static readonly TestCase[] TrainData;        
        public static readonly TestCase[] TestData;

        public static readonly int TrainCount;
        public static readonly int TestCount;

        public const int ImageSize = 784;
        public const int LabelSize = 10;


        static DataDb()
        {
            string dir = Directory.GetCurrentDirectory();
            string appFolder = dir.Split("/bin")[0];
            const string mnistPath = "../../data/MNIST";
            
            IEnumerable<TestCase>? trainDb = FileReaderMNIST.LoadImagesAndLables(Path.Combine(appFolder, $"{mnistPath}/train-labels-idx1-ubyte.gz"), Path.Combine(appFolder, $"{mnistPath}/train-images-idx3-ubyte.gz"));
            IEnumerable<TestCase>? testDb = FileReaderMNIST.LoadImagesAndLables(Path.Combine(appFolder, $"{mnistPath}/t10k-labels-idx1-ubyte.gz"), Path.Combine(appFolder, $"{mnistPath}/t10k-images-idx3-ubyte.gz"));

            TrainData = trainDb.ToArray();
            TestData = testDb.ToArray();

            TrainCount = TrainData.Length;
            TestCount = TestData.Length;
        }
    }
}