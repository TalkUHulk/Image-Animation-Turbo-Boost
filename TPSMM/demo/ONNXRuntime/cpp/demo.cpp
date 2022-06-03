#include <iostream>
#include "tpsmm.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <sys/types.h>
#endif


int main(int argc, char* argv[]) {
    if (argc != 6) {
            std::cout << "Usage : " << argv[0] << " <tpsmm_path_to_model> <kp_path_to_model> <path_to_source> <path_to_driving> <path_to_output>" << std::endl;
            return -1;
    }

    TPSMM* pModel = TPSMM::get_instance();
    pModel->init(argv[1],
                 argv[2]);

    auto source = cv::imread(argv[3]);
    cv::resize(source, source, cv::Size(256, 256));
    cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
    source.convertTo(source, CV_32F, 1.0 / 255.0);

    cv::Mat preprocessedSource;
    cv::dnn::blobFromImage(source, preprocessedSource);

    std::vector<float> kp_source;
    pModel->kp_detector(preprocessedSource, kp_source);

    auto cap = cv::VideoCapture(argv[4]);
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    auto video_writer = cv::VideoWriter(argv[5], codec, 25, cv::Size(256, 256));

    cv::Mat driving;

    while(true){
        cap >> driving;
        if(driving.empty()){
            break;
        }

        cv::Mat generated;

        cv::resize(driving, driving, cv::Size(256, 256));
        cv::cvtColor(driving, driving, cv::COLOR_BGR2RGB);

        driving.convertTo(driving, CV_32F, 1.0 / 255.0);

        cv::Mat preprocessedDriving;
        cv::dnn::blobFromImage(driving, preprocessedDriving);

        auto tic = std::chrono::high_resolution_clock::now();

        pModel->forward(kp_source, preprocessedSource, preprocessedDriving, generated);

        auto toc = std::chrono::high_resolution_clock::now();    //结束时间
        std::chrono::duration<double> elapsed = toc - tic;
        std::cout << "Elapsed time:" << elapsed.count() << "s" << std::endl;

        generated.convertTo(generated, CV_8U,  255);
        cv::cvtColor(generated, generated, cv::COLOR_BGR2RGB);

        video_writer << generated;

    }
    video_writer.release();
    cap.release();
    printf("Done!\n");
    return 0;

}
