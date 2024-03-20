//Задание 1
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

void contour(Mat& matrix, Mat& Matrix) {
    Mat img;
    GaussianBlur(matrix, img, Size(0, 0), 1.9);
    cvtColor(img, img, COLOR_BGR2GRAY);
    Matrix = Mat(matrix.rows, matrix.cols, CV_8U);

    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            float dx = img.at<uchar>(i + 1, j + 1) + 2 * img.at<uchar>(i, j + 1) + img.at<uchar>(i - 1, j + 1) - img.at<uchar>(i + 1, j - 1) - 2 * img.at<uchar>(i, j - 1) - img.at<uchar>(i - 1, j - 1);
            float dy = img.at<uchar>(i + 1, j + 1) + 2 * img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j - 1) - img.at<uchar>(i - 1, j - 1) - 2 * img.at<uchar>(i - 1, j) - img.at<uchar>(i - 1, j + 1);
            Matrix.at<uchar>(i, j) = 255 - sqrt(pow(dx, 2) + pow(dy, 2));
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    Mat image = imread("C:/Users/James-Bond/Downloads/mat.jpg");

    if (image.empty()) {
        cout << "ошибка загрузки картинки" << endl;
        return -1;
    }

    Mat grayImage, sepiaImage, negativeImage, contourImage;

    grayImage = image.clone();
    sepiaImage = image.clone();
    negativeImage = image.clone();
    contourImage = image.clone();

    omp_set_num_threads(5);

#pragma omp parallel for
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b colors = image.at<Vec3b>(i, j);
            uchar gray = colors[0] * 0.114 + colors[1] * 0.587 + colors[2] * 0.299;
            grayImage.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
        }
    }

#pragma omp parallel for
    Mat kern = (cv::Mat_<float>(4, 4) << 0.272, 0.534, 0.131, 0,
        0.349, 0.686, 0.168, 0,
        0.393, 0.769, 0.189, 0,
        0, 0, 0, 1);

        transform(sepiaImage, sepiaImage, kern);

#pragma omp parallel for
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b colors = image.at<Vec3b>(i, j);
            negativeImage.at<Vec3b>(i, j) = Vec3b(255 - colors[0], 255 - colors[1], 255 - colors[2]);
        }
    }

#pragma omp parallel for
    contour(image, contourImage);
    
    namedWindow("картинка", WINDOW_NORMAL);
    namedWindow("серый", WINDOW_NORMAL);
    namedWindow("сепия", WINDOW_NORMAL);
    namedWindow("негатив", WINDOW_NORMAL);
    namedWindow("контур", WINDOW_NORMAL);

    resizeWindow("картинка", 500, 281);
    resizeWindow("серый", 500, 281);
    resizeWindow("сепия", 500, 281);
    resizeWindow("негатив", 500, 281);
    resizeWindow("контур", 500, 281);

    imshow("картинка", image);
    imshow("серый", grayImage);
    imshow("сепия", sepiaImage);
    imshow("негатив", negativeImage);
    imshow("контур", contourImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
