namespace NeuralNetwork;

public static class Utility
{
	static double Sigmoid(double x)
	{
		return 1 / (1 + Math.Exp(-x));	
	}

	static double SigmoidDerivative(double x)
	{
		return Sigmoid(x) * (1 - Sigmoid(x));
	}
	
	static double ReLu(double x)
	{
		return double.Max(0, x);
	}

	static double ReLuDerivate(double x)
	{
		return x > 0 ? 1 : 0;
	}
	
	static double LeakyReLu(double x, double alpha)
	{
		return double.Max(alpha * x, x);
	}

	static double LeakyReLuDerivate(double x, double alpha)
	{
		return x > 0 ? 1 : alpha;
	}

	static double SoftMax(List<double> values, int index)
	{
		return Math.Exp(values[index]) / values.Sum(Math.Exp);
	}
}