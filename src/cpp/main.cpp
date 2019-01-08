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

extern void calc_opticalflow(string input_file_path, string output_stats_path, string output_movie_path);

int main(int argc, char **argv) {
    string input_file_path;
    string output_stats_dircpath;
    string is_saved_movie = "0";
    string output_movie_path = "";

    // receive processing file path from standard input
    cout << "input movie file path: ";
    cin >> input_file_path;
    cout << "input the output statistics directory: ";
    cin >> output_stats_dircpath;
    cout << "save optical flow movie (0:NO, 1:YES)";
    cin >> is_saved_movie;
    if(stoi(is_saved_movie)) {
        cout << "input the output file path: ";
        cin >> output_movie_path;
    }else{
        cout << "Not save output movie." << endl;
    }

    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    
    // calculate optical flow by above parameters
    calc_opticalflow(input_file_path, output_stats_dircpath, output_movie_path);

	end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast <chrono::seconds>(end - start).count();
    cout << "time: " << elapsed << " sec." << endl;

    return 0;
}
