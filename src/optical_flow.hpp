#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

using cv::VideoCapture;
using std::string;
using std::tuple;
using std::vector;

typedef cv::Point2f Pixel;
typedef tuple<vector<float>, vector<float>, vector<float>> StatsResultTuple;

class OpticalFlow {
 public:
  string input_video_path;
  string output_stats_dir;
  string output_video_path;
  VideoCapture capture;
  int width;
  int height;
  int total_frame;
  int fourcc;
  double fps;

  OpticalFlow(string input_video_path, string output_stats_dir,
              string output_video_path);

  // calculate the optical flow of each feature point between two frames
  vector<Pixel> calc_flow(vector<Pixel>& prev_corners,
                          vector<Pixel>& curr_corners, vector<uchar>& status);

  // calculate each norm by receiveing a vector of flow
  vector<float> calc_norm(vector<Pixel>& flow);

  // calculate optical flow of the input video
  StatsResultTuple calc_optical_flow();

  // save statistics computed by tracking.
  void save_stats_results(StatsResultTuple& status_results);
};