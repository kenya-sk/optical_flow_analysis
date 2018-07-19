#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctype.h>

using namespace std;

typedef cv::Point2f Pixel;

cv::Mat read_mask_image(string maskPath) {
    cv::Mat mask = cv::imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);
    if (mask.empty()) {
        cout << "Error: can not open image file. \n"
             << "PATH: " << maskPath << endl;
        exit(1);
    }

    return mask;
}


// split "string type" string with / to get file name
string StringSplit(string &str, char sep) {
    vector<string> sepVector;
    istringstream stream(str);
    string buffer;
    while (getline(stream, buffer, sep)) {
        sepVector.push_back(buffer);
    }
    string fileName = sepVector.back();

    return fileName;
}


// make input_file_path from outputFilePath　ex) ../201704280900.mp4 -> ../output/out_201704280900.mp4
string get_outputPath(string input_file_path) {
    string outputFilePath;
    string target = StringSplit(input_file_path, '/');
    string replacement = "output/out_" + target;
    if (!target.empty()) {
        string::size_type pos = 0;
        while ((pos = input_file_path.find(target, pos)) != string::npos) {
            outputFilePath = input_file_path.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }
    }

    return outputFilePath;
}


// make csv file by vector data and absolute path of output destination
void make_csv(vector<float> &vecData, string outputFilePath) {
    ofstream ofs(outputFilePath);
    if (ofs) {
        for (unsigned int i = 0; i < vecData.size(); i++) {
            ofs << vecData[i] << "\n";
        }
    }
    else {
        cout << "can not open file" << endl;
        exit(1);
    }
    ofs.close();
    cout << "out put csv file: " << outputFilePath << endl;
}


// calculate variance by receiving vector and average value
float calc_var(vector<float> &value, float mean) {
    float var = 0.0;
    for (int i = 0; i < value.size(); i++) {
        var += (value[i] - mean) * (value[i] - mean);
    }

    return var / value.size();
}


// calculate area ratio of specific area
float calc_area_ratio(cv::Mat &img, cv::Mat &binMaskImg) {
    cv::Mat tmpImg = img.clone();
    if (tmpImg.channels() != 1) {
        cv::cvtColor(tmpImg, tmpImg, CV_RGB2GRAY);
    }

    // binarizes the input image with the threshold value 150 and extracts only mask region
    cv::threshold(tmpImg, tmpImg, 150, 255, cv::THRESH_BINARY);

    int blackNum = 0, total = 0;
    for (int y = 0; y < tmpImg.rows; y++) {
        for (int x = 0; x < tmpImg.cols; x++) {
            int p1 = binMaskImg.at<uchar>(y, x);
            if (p1 == 1) {
                total++;
                int p2 = tmpImg.at<uchar>(y, x);
                if (p2 == 0) {
                    blackNum++;
                }
            }
        }
    }

    float ratio = float(blackNum) / total;

    return ratio;
}