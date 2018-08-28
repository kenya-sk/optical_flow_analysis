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

extern cv::Mat read_mask_image(string mask_path);
extern string StringSplit(string &str, char sep);
extern string get_outputPath(string input_file_path);
extern void make_csv(vector<float> &data_vec, string output_file_path);
extern float calc_var(vector<float> &value, float mean);
extern float calc_area_ratio(cv::Mat &img, cv::Mat &bin_mask_img);

// calculate the optical flow of each feature point between two frames
vector<Pixel> calc_flow(vector<Pixel> &prev_corners, vector<Pixel> &curr_corners, vector<uchar> &status) {
    vector<Pixel> tmp_flow;
    for (unsigned int i = 0; i < prev_corners.size(); i++) {
        if (status[i] == 1) {
            tmp_flow.push_back(curr_corners[i] - prev_corners[i]);
        }
    }

    return tmp_flow;
}

// calculate each norm by receiveing a vector of flow
vector<float> calc_norm(vector<Pixel> &flow)
{
    float pow_norm = 0.0;
    vector<float> flow_norm;
    for (unsigned int i = 0; i < flow.size(); i++)
    {
        pow_norm = (flow[i].x * flow[i].x) + (flow[i].y * flow[i].y);
        flow_norm.push_back(sqrt(pow_norm));
    }
    assert(flow.size() == flow_norm.size());

    return flow_norm;
}

void calc_opticalflow(string input_file_path, string output_stats_dircpath, string output_movie_path) {
    cv::VideoCapture capture(input_file_path);
    int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int total_frame = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fourcc = (int)capture.get(CV_CAP_PROP_FOURCC);
    double fps = (double)capture.get(CV_CAP_PROP_FPS);
    int fps_interval = 2; // spaceing to save frames

    // end if video can not be read
    if (!capture.isOpened()) {
        cout << "Error: can not open movie file." << endl;
        exit(1);
    }

    // display information of input file
    std::cout << "\n*******************************************" << std::endl;
    std::cout << "MOVIE PATH: " << input_file_path << std::endl;
    std::cout << "WIDTH: " << width << std::endl;
    std::cout << "HEIGHT: " << height << std::endl;
    std::cout << "TOTAL FRAME: " << total_frame << std::endl;
    std::cout << "FOURCC: " << fourcc << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "*******************************************\n" << std::endl;

    // set output file
    cv::VideoWriter writer;
    if (!output_movie_path.empty()) {
        writer = cv::VideoWriter(output_movie_path, fourcc, fps / fps_interval, cv::Size(width, height), true);
        cout << "output file path: " << output_movie_path << endl;
    }

    // end if mask image can not be read
    // aquarium area extraction mask
    cv::Mat aqua_mask = read_mask_image("/Users/sakka/optical_flow_analysis/image/mask.png");
    // human area extraction mask
    cv::Mat human_mask = cv::imread("/Users/sakka/optical_flow_analysis/image/human_mask.png");
    cv::Mat bin_human_mask = read_mask_image("/Users/sakka/optical_flow_analysis/image/bin_human_mask.png");

    cv::Mat frame, prev_gray, curr_gray;
    // store feature points of previous and next frames
    vector<Pixel> prev_corners, curr_corners;
    // whether correspondence of each feature point was found between two frames
    // 0:false 1:true
    vector<uchar> status;
    // represents the difference between the feature points
    // before and after the movement region
    vector<float> error;
    int window_size = ceil(fps / fps_interval);
    // retain value for window_size
    deque<float> tmp_mean_deq(window_size - 1, 0.0), tmp_var_deq(window_size - 1, 0.0), tmp_max_deq(window_size - 1, 0.0);
    vector<float> flow_norm, mean_vec, var_vec, max_vec, human_vec;
    vector<Pixel> flow;
    float flow_mean = 0.0, flow_var = 0.0, flow_max = 0.0;
    int frame_num = 0;

    while (true) {
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        frame_num++;
        if (frame_num % 100 == 0) {
            cout << "frame number: " << frame_num << "/" << total_frame << endl;
        }

        if (frame_num % fps_interval == 0){
            cv::cvtColor(frame, curr_gray, CV_RGB2GRAY);
            if (!prev_gray.empty()) {
                // extraction of feature points
                cv::goodFeaturesToTrack(prev_gray, prev_corners, 150, 0.2, 5, aqua_mask);
                // compute the optical flow and calculate the size(flow_norm)
                // only when the corresponding feature points is found
                cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners, curr_corners, status, error, cv::Size(20, 20), 5);
                flow = calc_flow(prev_corners, curr_corners, status);
                flow_norm = calc_norm(flow);
                // calculate mean, variance and maximum of optical flow
                flow_mean = accumulate(begin(flow_norm), end(flow_norm), 0.0) / flow_norm.size();
                flow_var = calc_var(flow_norm, flow_mean);
                if (flow_norm.size() > 0) {
                    flow_max = *max_element(flow_norm.begin(), flow_norm.end()); //Error
                }
                else {
                    flow_max = 0.0;
                }
                // disorder of video is detected based on the value of dispersion
                if (flow_var > 200) {
                    cout << "variance: " << flow_var << endl;
                    flow_var = 0.0;
                    flow_mean = 0.0;
                    flow_max = 0.0;
                }

                tmp_mean_deq.push_back(flow_mean);
                tmp_var_deq.push_back(flow_var);
                tmp_max_deq.push_back(flow_max);
                assert(tmp_mean_deq.size() == window_size);
                assert(tmp_var_deq.size() == window_size);
                assert(tmp_max_deq.size() == window_size);

                mean_vec.push_back(std::accumulate(tmp_mean_deq.begin(), tmp_mean_deq.end(), 0.0));
                var_vec.push_back(std::accumulate(tmp_var_deq.begin(), tmp_var_deq.end(), 0.0));
                max_vec.push_back(std::accumulate(tmp_max_deq.begin(), tmp_max_deq.end(), 0.0));

                tmp_mean_deq.pop_front();
                tmp_var_deq.pop_front();
                tmp_max_deq.pop_front();
                assert(tmp_mean_deq.size() == window_size - 1);
                assert(tmp_var_deq.size() == window_size - 1);
                assert(tmp_max_deq.size() == window_size - 1);

                // calculate area ratio of human area
                float ratio = calc_area_ratio(frame, bin_human_mask);
                human_vec.push_back(ratio);

                // write optical flow to the image
                if (!output_movie_path.empty()) {
                    for (unsigned int i = 0; i < curr_corners.size(); i++) {
                        if (status[i] == 1) {
                            cv::circle(frame, prev_corners[i], 3, cv::Scalar(0, 0, 255), -1, CV_AA);
                            cv::line(frame, prev_corners[i], curr_corners[i], cv::Scalar(0, 0, 255), 1, CV_AA);
                        }
                    }
                    writer << frame;
                }
            }
            prev_gray = curr_gray.clone();
        }
    }
    cv::destroyAllWindows();

    string file_name;
    file_name = StringSplit(input_file_path, '/');
    // excluding ".mp4" from fileName
    file_name.erase(file_name.end() - 4, file_name.end());

    make_csv(mean_vec, output_stats_dircpath+"/mean.csv");
    make_csv(var_vec, output_stats_dircpath+"var.csv");
    make_csv(max_vec, output_stats_dircpath+"/max.csv");
    make_csv(human_vec, output_stats_dircpath+"/human.csv");
}
