#include "utility.h"

double** read_mnist_images(string full_path, int& number_of_images, int& image_size);
double** read_mnist_labels(string full_path, int& img_num);

double** getTrainingImages(int& number_of_images, int& image_size);
double** getTestingImages(int& number_of_images, int& image_size);
double** getTrainingLabels(int& img_num);
double** getTestingLabels(int& img_num);