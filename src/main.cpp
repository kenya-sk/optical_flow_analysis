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
#include <tuple>
#include <vector>
#include "optical_flow.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::tuple;

using namespace std::chrono;

typedef cv::Point2f Pixel;
typedef tuple<vector<float>, vector<float>, vector<float>> StatsResultTuple;

int main(int argc, char** argv) {
  string input_video_path;
  string output_stats_dir;
  string is_saved_video = "0";
  string output_video_path = "";

  // receive processing file path from standard input
  cout << "input video file path: ";
  cin >> input_video_path;
  cout << "input the output statistics directory: ";
  cin >> output_stats_dir;
  cout << "save optical flow video (0:NO, 1:YES)";
  cin >> is_saved_video;
  if (stoi(is_saved_video)) {
    cout << "input the output file path: ";
    cin >> output_video_path;
  } else {
    cout << "Not save output video." << endl;
  }

  system_clock::time_point start, end;
  start = system_clock::now();

  OpticalFlow tracker =
      OpticalFlow(input_video_path, output_stats_dir, output_video_path);

  // calculate optical flow by above parameters and save results
  StatsResultTuple result_tuple = tracker.calc_optical_flow();
  tracker.save_stats_results(result_tuple);

  end = system_clock::now();
  double elapsed = duration_cast<seconds>(end - start).count();
  cout << "time: " << elapsed << " sec." << endl;

  return 0;
}
