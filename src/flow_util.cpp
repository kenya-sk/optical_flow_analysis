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

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::istringstream;
using std::ofstream;


typedef cv::Point2f Pixel;

void pretty_print(string input_file_path, int width, int height, int total_frame, int fourcc, double fps) {
    /*
    display information of input file.
    */

    std::cout << "\n*******************************************" << std::endl;
    std::cout << "VIDEO PATH: " << input_file_path << std::endl;
    std::cout << "WIDTH: " << width << std::endl;
    std::cout << "HEIGHT: " << height << std::endl;
    std::cout << "TOTAL FRAME: " << total_frame << std::endl;
    std::cout << "FOURCC: " << fourcc << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "******************************************* \n" << std::endl;
}


cv::Mat read_mask_image(string mask_path) {
    /*
    read binary mask image.
    mask image defined bellow:
        not calculated pixel = 0
        calculated pixel = 1 or 255
    */

    cv::Mat mask = cv::imread(mask_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (mask.empty()) {
        cout << "Error: can not open image file. \n"
             << "PATH: " << mask_path << endl;
        exit(1);
    }

    return mask;
}


string string_split(string &str, char sep) {
    /*
    split "string type" string with "sep" to get file name
    */

    vector<string> sep_vec;
    istringstream stream(str);
    string buffer;
    while (getline(stream, buffer, sep)) {
        sep_vec.push_back(buffer);
    }
    string file_name = sep_vec.back();

    return file_name;
}


string get_outputPath(string input_file_path) {
    /*
    make input_file_path from output_file_path
    ex) ../201704280900.mp4 -> ../output/out_201704280900.mp4
    */

    string output_file_path;
    string target = string_split(input_file_path, '/');
    string replacement = "output/out_" + target;
    if (!target.empty()) {
        string::size_type pos = 0;
        while ((pos = input_file_path.find(target, pos)) != string::npos) {
            output_file_path = input_file_path.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }
    }

    return output_file_path;
}


void make_csv(vector<float> &data_vec, string output_file_path) {
    /*
    make csv file by vector data and absolute path of output destination
    */

    ofstream ofs(output_file_path);
    if (ofs) {
        for (unsigned int i = 0; i < data_vec.size(); i++) {
            ofs << data_vec[i] << "\n";
        }
    }
    else {
        cout << "can not open file" << endl;
        exit(1);
    }
    ofs.close();
    cout << "output csv file: " << output_file_path << endl;
}


float calc_var(vector<float> &value, float mean) {
    /*
    calculate variance by receiving vector and average value
    */
    float var = 0.0;
    for (int i = 0; i < value.size(); i++) {
        var += (value[i] - mean) * (value[i] - mean);
    }

    return var / value.size();
}


float calc_area_ratio(cv::Mat &img, cv::Mat &bin_mask_img) {
    /*
    calculate area ratio of specific area
    */

    cv::Mat tmp_img = img.clone();
    if (tmp_img.channels() != 1) {
        cv::cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);
    }

    // binarizes the input image with the threshold value 150 and extracts only mask region
    cv::threshold(tmp_img, tmp_img, 150, 255, cv::THRESH_BINARY);

    int black_num = 0, total = 0;
    for (int y = 0; y < tmp_img.rows; y++) {
        for (int x = 0; x < tmp_img.cols; x++) {
            int p1 = bin_mask_img.at<uchar>(y, x);
            if (p1 == 1) {
                total++;
                int p2 = tmp_img.at<uchar>(y, x);
                if (p2 == 0) {
                    black_num++;
                }
            }
        }
    }

    float ratio = float(black_num) / total;

    return ratio;
}