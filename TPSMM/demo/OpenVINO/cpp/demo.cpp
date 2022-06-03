#include <iostream>
#include "tpsmm.hpp"

int main(int argc, char* argv[]) {

    if (argc != 8) {
            std::cout << "Usage : " << argv[0] << " <tpsmm_path_to_xml> <tpsmm_path_to_bin> <kp_path_to_xml> <kp_path_to_bin> <path_to_source> <path_to_driving> <path_to_output>" << std::endl;
            return -1;
    }

    TPSMM* pModel = TPSMM::get_instance();
    pModel->init(argv[1], argv[2],
                 argv[3], argv[4]);

    auto source = cv::imread(argv[5]);
    cv::resize(source, source, cv::Size(256, 256));
    cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
    source.convertTo(source, CV_32F, 1.0 / 255.0);

    InferenceEngine::Blob::Ptr kp_source;
    pModel->kp_detector(source, kp_source);
    std::cout << "kp_source size:" << kp_source->size() << std::endl;

    auto cap = cv::VideoCapture(argv[6]);

    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    auto video_writer = cv::VideoWriter(argv[7], codec, 25, cv::Size(256, 256));

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


        auto tic = std::chrono::high_resolution_clock::now();

        pModel->forward(kp_source, source, driving, generated);

        auto toc = std::chrono::high_resolution_clock::now();
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
