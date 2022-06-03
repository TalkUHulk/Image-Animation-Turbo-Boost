//
// Created by TalkUHulk on 2022/5/31.
//

#ifndef DEMO_TPSMM_HPP
#define DEMO_TPSMM_HPP
#include <iostream>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <sys/types.h>
#endif

class TPSMM{
private:
    Ort::Session* _session=nullptr;
    Ort::Session* _session_kp_detector=nullptr;
    std::vector<const char*> _input_node_names = {"kp_source", "source", "driving"};
    std::vector<const char*> _output_node_names = {"output"};
    std::vector<const char*> _kp_input_node_names = {"source"};
    std::vector<const char*> _kp_output_node_names = {"kp_source"};
    Ort::Env _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TPSMM");
    Ort::SessionOptions _session_options = Ort::SessionOptions();

    std::vector<int64_t> _kp_dims = {1, 50, 2};
    size_t _kp_size = 100;

    std::vector<int64_t> _input_dims = {1, 3, 256, 256};
    size_t _input_size = 1 * 3 * 256 * 256;
    OrtMemoryInfo* _memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    TPSMM() =  default;
public:
    static TPSMM* get_instance(){
        static TPSMM* _static_detector_instance;
        if (nullptr == _static_detector_instance){
            _static_detector_instance = new TPSMM();
        }
        return _static_detector_instance;
    }

    ~TPSMM(){
        if(this->_session != nullptr){
            delete this->_session;
            this->_session = nullptr;
        }
        if(this->_session_kp_detector != nullptr){
            delete this->_session_kp_detector;
            this->_session_kp_detector = nullptr;
        }
    }

    void init(const char* tpsmm_model_path, const char* kp_detector_model_path, bool m_isGPU);
    void init(const char* tpsmm_model_path, const char* kp_detector_model_path);
    void forward(const std::vector<float>& kp_source, const cv::Mat& source, const cv::Mat& driving, cv::Mat& output);
    void kp_detector(const cv::Mat& input, std::vector<float>& output);
};
#endif //DEMO_TPSMM_HPP
