#include "mnist.h"

int reverseInt(unsigned int num)
{
    return __builtin_bswap32(num);
}

double** read_mnist_images(string full_path, int& img_num, int& image_size) 
{
    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        streamsize intSize = sizeof(int);
        int magic_number = 0, row_num = 0, col_num = 0;

        file.read((char*) &magic_number, intSize);
        magic_number = reverseInt(magic_number);
        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*) &img_num, intSize);
        file.read((char*) &row_num, intSize);
        file.read((char*) &col_num, intSize);
        
        img_num = reverseInt(img_num), row_num = reverseInt(row_num), col_num = reverseInt(col_num);
        image_size = row_num * col_num;

        double** dataSet = new double*[img_num];
        for(int i = 0; i < img_num; i++) {
            uchar* image = new uchar[image_size];
            dataSet[i] = new double[image_size];

            file.read((char *)image, image_size);
            for(int j=0; j<image_size; j++)
            {
                dataSet[i][j] = int(image[j]) / 255.0;
            }
        }
        return dataSet;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

double** read_mnist_labels(string full_path, int& img_num) 
{
    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        streamsize intSize = sizeof(int);
        streamsize labelSize = 1;
        int magic_number = 0;

        file.read((char*) &magic_number, intSize);
        magic_number = reverseInt(magic_number);
        if(magic_number != 2049) throw runtime_error("Invalid MNIST image file!");

        file.read((char*) &img_num, intSize);
        img_num = reverseInt(img_num);

        double** dataSet = new double*[img_num];
        for(int i = 0; i < img_num; i++) {
            int number;
            file.read((char *) &number, labelSize);
            cout << number << endl;
            dataSet[i] = new double[10];
            for(int j=0;j<10;j++)
            {
                if(j == number) dataSet[i][j] = 1.0;
                else dataSet[i][j] = 0.0;
            }
        }
        return dataSet;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

double** getTrainingImages(int& img_num, int& image_size)
{
    string path = "/home/gwynn7/Documents/Programming/NeuralNetwork/data/MNIST/train-images-idx3-ubyte";
    return read_mnist_images(path, img_num, image_size);
}

double** getTestingImages(int& img_num, int& image_size)
{
    string path = "/home/gwynn7/Documents/Programming/NeuralNetwork/data/MNIST/t10k-images-idx3-ubyte";
    return read_mnist_images(path, img_num, image_size);
}

double** getTrainingLabels(int& img_num)
{
    string path = "/home/gwynn7/Documents/Programming/NeuralNetwork/data/MNIST/train-labels-idx1-ubyte";
    return read_mnist_labels(path, img_num);
}

double** getTestingLabels(int& img_num)
{
    string path = "/home/gwynn7/Documents/Programming/NeuralNetwork/data/MNIST/t10k-labels-idx1-ubyte";
    return read_mnist_labels(path, img_num);
}
