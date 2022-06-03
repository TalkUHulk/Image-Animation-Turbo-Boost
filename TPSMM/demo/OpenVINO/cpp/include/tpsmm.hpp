//
// Created by TalkUHulk on 2022/6/2.
//

#ifndef OPENVINO_TPSMM_H
#define OPENVINO_TPSMM_H
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

class TPSMM{
private:
    InferenceEngine::Core _ie_core_kp;
    InferenceEngine::CNNNetwork _network_kp;
    InferenceEngine::ExecutableNetwork _executable_network_kp;
    InferenceEngine::InferRequest _infer_request_kp;

    InferenceEngine::Core _ie_core;
    InferenceEngine::CNNNetwork _network;
    InferenceEngine::ExecutableNetwork _executable_network;
    InferenceEngine::InferRequest _infer_request;

    TPSMM() =  default;
public:
    static TPSMM* get_instance(){
        static TPSMM* _static_detector_instance;
        if (nullptr == _static_detector_instance){
            _static_detector_instance = new TPSMM();
        }
        return _static_detector_instance;
    }

    ~TPSMM(){}

    void init(const std::string& modelPath, const std::string& binPath,
              const std::string& kp_modelPath, const std::string& kp_binPath,
              const std::string& deviceName="CPU");
    void kp_detector(const cv::Mat& input, InferenceEngine::Blob::Ptr &output);
    void forward(const InferenceEngine::Blob::Ptr &kp_source, const cv::Mat& source, const cv::Mat& driving, cv::Mat& output);
};

#endif //OPENVINO_TPSMM_H
