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

cv::Mat read_mask_image(string maskPath){
    cv::Mat mask = cv::imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);
    if (mask.empty()){
        cout << "Error: can not open image file. \n" << "PATH: " << maskPath << endl;
        exit(1);
    }

    return mask;
}


// split "string type" string with / to get file name
string StringSplit(string &str, char sep){
    vector<string> sepVector;
    istringstream stream(str);
    string buffer;
    while(getline(stream, buffer, sep)){
        sepVector.push_back(buffer);
    }
    string fileName = sepVector.back();

    return fileName;
}


// make inputFilePath from outputFilePath　ex) ../201704280900.mp4 -> ../out_201704280900.mp4
string get_outputPath(string inputFilePath){
    string outputFilePath;
    string target = StringSplit(inputFilePath, '/');
    string replacement = "output/out_" + target;
    if (!target.empty()){
        string::size_type pos = 0;
        while((pos = inputFilePath.find(target, pos)) != string::npos){
            outputFilePath =  inputFilePath.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }
    }

    return outputFilePath;
}


// calculate the optical flow of each feature point between two frames
vector<Pixel> calc_flow(vector<Pixel> &prevCorners, vector<Pixel> &currCorners, vector<uchar> &status){
    vector<Pixel> tmpFlow;
    for (unsigned int i = 0; i < prevCorners.size(); i++){
        if (status[i] == 1){
            tmpFlow.push_back(currCorners[i] - prevCorners[i]);
        }
    }

    return tmpFlow;
}


// calculate each norm by receiveing a vector of flow
vector<float> calc_norm(vector<Pixel> &flow){
    float powNorm = 0.0;
    vector<float> flowNorm;
    for (unsigned  int i = 0; i < flow.size(); i++){
        powNorm = (flow[i].x * flow[i].x) + (flow[i].y * flow[i].y);
        flowNorm.push_back(sqrt(powNorm));
    }
    assert(flow.size() == flowNorm.size());

    return flowNorm;
}


// calculate variance by receiving vector and average value
float calc_var(vector<float> &value, float mean){
    float var = 0.0;
    for (int i = 0; i < value.size(); i++){
        var += (value[i] - mean) * (value[i] - mean);
    }

    return var / value.size();
}


// calculate area ratio of specific area
float calc_area_ratio(cv::Mat &img, cv::Mat &binMaskImg){
    cv::Mat tmpImg = img.clone();
    if (tmpImg.channels() != 1){
        cv::cvtColor(tmpImg, tmpImg, CV_RGB2GRAY);
    }

    // binarizes the input image with the threshold value 150 and extracts only mask region
    cv::threshold(tmpImg, tmpImg, 150, 255, cv::THRESH_BINARY);

    int blackNum = 0, total = 0;
    for (int y=0; y < tmpImg.rows; y++){
        for (int x=0; x < tmpImg.cols; x++){
            int p1 = binMaskImg.at<uchar>(y, x);
            if (p1 == 1){
                total++;
                int p2 = tmpImg.at<uchar>(y, x);
                if (p2 == 0){
                    blackNum++;
                }
            }
        }
    }

    float ratio = float(blackNum) / total;

    return ratio;
}


// make csv file by vector data and absolute path of output destination
void make_csv(vector<float> &vecData, string outputFilePath){
    ofstream ofs(outputFilePath);
    if (ofs){
        for (unsigned int i = 0; i < vecData.size(); i++){
            ofs << vecData[i] << "\n";
        }
    }else{
        cout << "can not open file" << endl;
        exit(1);
    }
    ofs.close();
    cout << "out put csv file: " << outputFilePath << endl;
}


void calc_opticalflow(string inputFilePath, bool output){
    cv::VideoCapture capture(inputFilePath);
    int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int count = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fourcc = (int)capture.get(CV_CAP_PROP_FOURCC);
    double fps = capture.get(CV_CAP_PROP_FPS);
    int fpsInterval = 2; // spaceing to save frames

    // end if video can not be read
    if (!capture.isOpened()){
        cout << "Error: can not open movie file." << endl;
        exit(1);
    }

    // display information of input file
    cout << "input file path: " << inputFilePath << endl;
    cout << "\nwidth: " << width << "\nheight: " << height << "\ncount: " << count << "\nfourcc: " << fourcc << "\nfps: " << fps << endl;

    // set output file
    cv::VideoWriter writer;
    if (output){
        string outputFilePath = get_outputPath(inputFilePath);
        writer = cv::VideoWriter(outputFilePath, fourcc, fps/fpsInterval, cv::Size(width, height), true);
        cout << "output file path: " << outputFilePath << endl;
    }

    // end if mask image can not be read
    // aquarium area extraction mask
    cv::Mat aquaMask = read_mask_image("/Users/sakka/FastAnomalyDetection/image/mask.png");
    // human area extraction mask
    cv::Mat humanMask = cv::imread("/Users/sakka/FastAnomalyDetection/image/humanMask.png");
    cv::Mat binHumanMask = read_mask_image("/Users/sakka/FastAnomalyDetection/image/binHumanMask.png");

    cv::Mat frame, prevGray, currGray, maskedGray;
    // store feature points of previous and next frames
    vector<Pixel> prevCorners, currCorners;
    // whether correspondence of each feature point was found between two frames
    // 0:false 1:true
    vector<uchar> status;
    // represents the difference between the feature points
    // before and after the movement region
    vector<float> error;
    int windowSize = ceil(fps/fpsInterval);
    // retain value for windowSize
    deque<float> tmpMeanDeq(windowSize - 1, 0.0), tmpVarDeq(windowSize - 1, 0.0), tmpMaxDeq(windowSize - 1, 0.0);
    vector<float> flowNorm, meanVec, varVec, maxVec, humanVec;
    vector<Pixel> prevCornersFiltered, currCornersFiltered, flow;
    float flowMean = 0.0, flowVar = 0.0, flowMax = 0.0;
    int frameNum = 0;

    while(true){
        capture >> frame;
        if (frame.empty()){
            break;
        }
        frameNum++;
        if (frameNum % 100 == 0) {
            cout << "frame number: " << frameNum << endl;
        }

        if(frameNum % fpsInterval == 0){
            cv::cvtColor(frame, currGray, CV_RGB2GRAY);
            if (!prevGray.empty()){
                // extraction of feature points
                cv::goodFeaturesToTrack(prevGray, prevCorners, 150, 0.2, 5, aquaMask);
                // compute the optical flow and calculate the size(flowNorm)
                // only when the corresponding feature points is found
                cv::calcOpticalFlowPyrLK(prevGray, currGray, prevCorners, currCorners, status, error, cv::Size(20,20), 5);
                flow = calc_flow(prevCorners, currCorners, status);
                flowNorm = calc_norm(flow);
                // calculate mean, variance and maximum of optical flow
                flowMean = accumulate(begin(flowNorm), end(flowNorm), 0.0) / flowNorm.size();
                flowVar = calc_var(flowNorm, flowMean);
                if (flowNorm.size() > 0){
                    flowMax = *max_element(flowNorm.begin(), flowNorm.end()); //Error
                }else{
                    flowMax = 0.0;
                }
                // disorder of video is detected based on the value of dispersion
                if (flowVar > 200) {
                    cout << "variance: " << flowVar << endl;
                    flowVar = 0.0;
                    flowMean = 0.0;
                    flowMax = 0.0;
                }

                tmpMeanDeq.push_back(flowMean);
                tmpVarDeq.push_back(flowVar);
                tmpMaxDeq.push_back(flowMax);
                assert (tmpMeanDeq.size() == windowSize);
                assert (tmpVarDeq.size() == windowSize);
                assert (tmpMaxDeq.size() == windowSize);

                meanVec.push_back(std::accumulate(tmpMeanDeq.begin(), tmpMeanDeq.end(), 0.0));
                varVec.push_back(std::accumulate(tmpVarDeq.begin(), tmpVarDeq.end(), 0.0));
                maxVec.push_back(std::accumulate(tmpMaxDeq.begin(), tmpMaxDeq.end(), 0.0));

                tmpMeanDeq.pop_front();
                tmpVarDeq.pop_front();
                tmpMaxDeq.pop_front();
                assert (tmpMeanDeq.size() == windowSize - 1);
                assert (tmpVarDeq.size() == windowSize - 1);
                assert (tmpMaxDeq.size() == windowSize - 1);

                // calculate area ratio of human area
                float ratio = calc_area_ratio(frame, binHumanMask);
                humanVec.push_back(ratio);

                // write optical flow to the image
                if (output){
                     for (unsigned int i = 0; i < currCorners.size(); i++){
                         if (status[i] == 1){
                             cv::circle(frame, prevCorners[i], 3, cv::Scalar(0,0,255), -1, CV_AA);
                             cv::line(frame, prevCorners[i], currCorners[i], cv::Scalar(0, 0, 255), 1, CV_AA);
                         }
                     }
                    writer << frame;
                }

            }
            prevGray = currGray.clone();
        }

    }
    cv::destroyAllWindows();

    string fileName;
    fileName = StringSplit(inputFilePath, '/');
    // excluding ".mp4" from fileName
    fileName.erase(fileName.end() - 4 ,fileName.end());

    make_csv(meanVec, "/Users/sakka/FastAnomalyDetection/data/2017-04-28/mean/mean_" + fileName + ".csv");
    make_csv(varVec, "/Users/sakka/FastAnomalyDetection/data/2017-04-28/var/var_" + fileName + ".csv");
    make_csv(maxVec, "/Users/sakka/FastAnomalyDetection/data/2017-04-28/max/max_" + fileName + ".csv");
    make_csv(humanVec, "/Users/sakka/FastAnomalyDetection/data/2017-04-28/human/human_" + fileName + ".csv");
}


int main (int argc, char **argv) {
	chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    string inputFilePath;

    // receive processing file path from standard input
    cin >> inputFilePath;
    calc_opticalflow(inputFilePath, true);
    cout << "end: " << inputFilePath << endl;


	end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast <chrono::seconds>(end - start).count();
    cout << "time: " << elapsed << " sec." << endl;

    return 0;
}
