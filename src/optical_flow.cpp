#include "optical_flow.hpp"
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>
#include <vector>
#include "utils.hpp"

using cv::Mat;
using std::cout;
using std::deque;
using std::endl;
using std::fixed;
using std::setprecision;
using std::string;
using std::tuple;
using std::vector;

typedef cv::Point2f Pixel;
typedef tuple<vector<float>, vector<float>, vector<float>> StatsResultTuple;

OpticalFlow::OpticalFlow(string input_video_path, string output_stats_dir,
                         string output_video_path) {
  // caputure the video.
  // if the video can not be open, it will end.
  capture = *(new cv::VideoCapture(input_video_path));
  if (!capture.isOpened()) {
    cout << "ERROR: can not open file (input video). please check file path."
         << endl;
    cout << "input path: " << input_video_path << endl;
    exit(1);
  }

  // set path
  this->input_video_path = input_video_path;
  this->output_stats_dir = output_stats_dir;
  this->output_video_path = output_video_path;

  // set video infomation
  width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  total_frame = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
  fourcc = (int)capture.get(cv::CAP_PROP_FOURCC);
  fps = (double)capture.get(cv::CAP_PROP_FPS);
}

vector<Pixel> OpticalFlow::calc_flow(vector<Pixel>& prev_corners,
                                     vector<Pixel>& curr_corners,
                                     vector<uchar>& status) {
  /*
  calculate the optical flow of each feature point between two frames
  */

  vector<Pixel> tmp_flow;
  for (unsigned int i = 0; i < prev_corners.size(); i++) {
    if (status[i] == 1) {
      tmp_flow.push_back(curr_corners[i] - prev_corners[i]);
    }
  }

  return tmp_flow;
}

vector<float> OpticalFlow::calc_norm(vector<Pixel>& flow) {
  /*
  calculate each norm by receiveing a vector of flow
  */

  float pow_norm = 0.0;
  vector<float> flow_norm;

  for (unsigned int i = 0; i < flow.size(); i++) {
    pow_norm = (flow[i].x * flow[i].x) + (flow[i].y * flow[i].y);
    flow_norm.push_back(sqrt(pow_norm));
  }
  assert(flow.size() == flow_norm.size());

  return flow_norm;
}

StatsResultTuple OpticalFlow::calc_optical_flow() {
  /*
  calculate optical flow of the input video
  */

  // display the information of input file
  pretty_print(input_video_path, width, height, total_frame, fourcc, fps);

  // set output file
  cv::VideoWriter writer;
  if (!output_video_path.empty()) {
    writer = cv::VideoWriter(output_video_path, fourcc, fps,
                             cv::Size(width, height), true);
    cout << "output file path: " << output_video_path << endl;
  }

  // Mask image for extracting the target area
  // Mat aqua_mask = read_mask_image(mask_path);
  Mat aqua_mask = Mat::ones(cv::Size(width, height), CV_8UC1);

  // frame data using optical flow
  Mat frame, prev_gray, curr_gray;

  // save the trajectory of tracking
  Mat tracking_mask = Mat::zeros(cv::Size(width, height), CV_8UC3);

  // store feature points of previous and next frames
  vector<Pixel> prev_corners, curr_corners;

  // whether correspondence of each feature point was found between two frames
  // 0:false 1:true
  vector<uchar> status;

  // represents the difference between the feature points
  // before and after the movement region
  vector<float> error;
  int window_size = ceil(fps);

  // retain value for window_size
  deque<float> tmp_mean_deq(window_size - 1, 0.0),
      tmp_var_deq(window_size - 1, 0.0), tmp_max_deq(window_size - 1, 0.0);
  vector<float> flow_norm, mean_vec, var_vec, max_vec;
  vector<Pixel> flow;
  float flow_mean = 0.0, flow_var = 0.0, flow_max = 0.0;
  int frame_num = 0;
  int feature_num = 100;

  cout << "\n*************** [Start]: Optical Flow Tracking ***************"
       << endl;
  while (true) {
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    frame_num++;
    // display progress
    if (frame_num % 100 == 0) {
      float percentage = 100 * (float)frame_num / (float)total_frame;
      cout << "progress: " << fixed << setprecision(2) << percentage << "% ["
           << frame_num << "/" << total_frame << "]" << endl;
    }

    // convert RGB frame to Gray scale
    cv::cvtColor(frame, curr_gray, cv::COLOR_RGB2GRAY);
    if (!prev_gray.empty()) {
      // extraction of feature points
      cv::goodFeaturesToTrack(prev_gray, prev_corners, feature_num, 0.2, 5,
                              aqua_mask);

      // compute the optical flow and calculate the size(flow_norm)
      // only when the corresponding feature points is found
      if (prev_corners.size() > 0) {
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners,
                                 curr_corners, status, error, cv::Size(20, 20),
                                 5);
        flow = calc_flow(prev_corners, curr_corners, status);
        flow_norm = calc_norm(flow);
      }

      // calculate mean, variance and maximum of optical flow
      flow_mean =
          accumulate(begin(flow_norm), end(flow_norm), 0.0) / flow_norm.size();
      flow_var = calc_var(flow_norm, flow_mean);
      if (flow_norm.size() > 0) {
        flow_max = *max_element(flow_norm.begin(), flow_norm.end());  // Error
      } else {
        flow_max = 0.0;
      }

      // save statistics value of current frame
      tmp_mean_deq.push_back(flow_mean);
      tmp_var_deq.push_back(flow_var);
      tmp_max_deq.push_back(flow_max);
      assert(tmp_mean_deq.size() == window_size);
      assert(tmp_var_deq.size() == window_size);
      assert(tmp_max_deq.size() == window_size);

      // cumulate window_size value
      mean_vec.push_back(
          std::accumulate(tmp_mean_deq.begin(), tmp_mean_deq.end(), 0.0));
      var_vec.push_back(
          std::accumulate(tmp_var_deq.begin(), tmp_var_deq.end(), 0.0));
      max_vec.push_back(
          std::accumulate(tmp_max_deq.begin(), tmp_max_deq.end(), 0.0));

      tmp_mean_deq.pop_front();
      tmp_var_deq.pop_front();
      tmp_max_deq.pop_front();
      assert(tmp_mean_deq.size() == window_size - 1);
      assert(tmp_var_deq.size() == window_size - 1);
      assert(tmp_max_deq.size() == window_size - 1);

      // write optical flow to the image
      if (!output_video_path.empty()) {
        // add the current trajectory to the past trajectory
        for (unsigned int i = 0; i < curr_corners.size(); i++) {
          if (status[i] == 1) {
            cv::line(tracking_mask, prev_corners[i], curr_corners[i],
                     cv::Scalar(0, 0, 255), 2, CV_AA);
          }
        }
        // only plot current feature point
        cv::add(frame, tracking_mask, frame);
        for (unsigned int i = 0; i < curr_corners.size(); i++) {
          if (status[i] == 1) {
            cv::circle(frame, curr_corners[i], 5, cv::Scalar(0, 0, 255), -1,
                       CV_AA);
          }
        }
        // save image trajectory
        writer << frame;
      }
    }

    // reset tracking mask for every 1 sec
    if (frame_num % int(fps) == 0) {
      tracking_mask = Mat::zeros(cv::Size(width, height), CV_8UC3);
    }

    prev_gray = curr_gray.clone();
  }
  cv::destroyAllWindows();

  cout << "\n*************** [End]: Optical Flow Tracking ***************"
       << endl;

  // combine computed statistics into a single tuple
  StatsResultTuple tracking_stats_tuple = std::tie(mean_vec, var_vec, max_vec);

  return tracking_stats_tuple;
}

void OpticalFlow::save_stats_results(StatsResultTuple& status_results) {
  /** save statistics computed by tracking
   **/
  int tuple_size = std::tuple_size<StatsResultTuple>::value;
  assert(tuple_size == 3 &&
         "The tuple size of the statistical results is expected to be 5.");

  // extract each result vector
  auto mean_vec = std::get<0>(status_results);
  auto var_vec = std::get<1>(status_results);
  auto max_vec = std::get<2>(status_results);

  cout << "\n*******************************************" << endl;
  // mean result
  string save_mean_path = output_stats_dir + "/mean.csv";
  make_csv(mean_vec, save_mean_path);

  // variance result
  string save_var_path = output_stats_dir + "/var.csv";
  make_csv(var_vec, save_var_path);

  // max result
  string save_max_path = output_stats_dir + "/max.csv";
  make_csv(max_vec, save_max_path);
  cout << "*******************************************" << endl;
}
