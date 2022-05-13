#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>

using cv::Mat;
using std::string;
using std::vector;

typedef cv::Point2f Pixel;

// display information of input file
void pretty_print(string input_file_path, int width, int height,
                  int total_frame, int fourcc, double fps);

// split "string type" string with "sep" to get file name
string string_split(string& str, char sep);

// make input_file_path from output_file_path
string get_outputPath(string input_file_path);

// make csv file by vector data and absolute path of output destination
void make_csv(vector<float>& data_vec, string output_file_path);

// calculate variance by receiving vector and average value
float calc_var(vector<float>& value, float mean);