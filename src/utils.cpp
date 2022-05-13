#include <ctype.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

using std::cout;
using std::endl;
using std::istringstream;
using std::ofstream;
using std::string;
using std::vector;

typedef cv::Point2f Pixel;

void pretty_print(string input_file_path, int width, int height,
                  int total_frame, int fourcc, double fps) {
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

string string_split(string& str, char sep) {
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
      output_file_path =
          input_file_path.replace(pos, target.length(), replacement);
      pos += replacement.length();
    }
  }

  return output_file_path;
}

void make_csv(vector<float>& data_vec, string output_file_path) {
  /*
  make csv file by vector data and absolute path of output destination
  */

  ofstream ofs(output_file_path);
  if (ofs) {
    for (unsigned int i = 0; i < data_vec.size(); i++) {
      ofs << data_vec[i] << "\n";
    }
  } else {
    cout << "can not open file" << endl;
    exit(1);
  }
  ofs.close();
  cout << "output csv file: " << output_file_path << endl;
}

float calc_var(vector<float>& value, float mean) {
  /*
  calculate variance by receiving vector and average value
  */
  float var = 0.0;
  for (int i = 0; i < value.size(); i++) {
    var += (value[i] - mean) * (value[i] - mean);
  }

  return var / value.size();
}