//
// Created by TalkUHulk on 2022/5/31.
//
#include "tpsmm.hpp"


const int CV_MAX_DIM = 32;
cv::Mat getPlane(const cv::Mat &m, int n, int cn)
{
    CV_Assert(m.dims > 2);
    int sz[CV_MAX_DIM];
    for(int i = 2; i < m.dims; i++)
    {
        sz[i-2] = m.size.p[i];
    }
    return cv::Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));

}

void imagesFromBlob(const cv::Mat& blob_, cv::OutputArrayOfArrays images_)
{
    //blob 是浮点精度的4维矩阵
    //blob_[0] = 批量大小 = 图像数
    //blob_[1] = 通道数
    //blob_[2] = 高度
    //blob_[3] = 宽度
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(blob_.size[2],blob_.size[3],blob_.depth() );//创建一个图像


    std::vector<cv::Mat> vectorOfChannels(blob_.size[1]);
    {int n = 0;
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_);
    }
}

void TPSMM::init(const char* tpsmm_model_path, const char* kp_detector_model_path, bool m_isGPU){

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (m_isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (m_isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        this->_session_options.AppendExecutionProvider_CUDA(cudaOption);// 加入CUDA
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    this->_session_options.SetIntraOpNumThreads(16);

    this->_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    this->_session_options.SetLogSeverityLevel(4);

    this->_session = new Ort::Session(this->_env, tpsmm_model_path, this->_session_options);
    this->_session_kp_detector = new Ort::Session(this->_env, kp_detector_model_path, this->_session_options);
}

void TPSMM::init(const char* tpsmm_model_path, const char* kp_detector_model_path){

    this->_session = new Ort::Session(this->_env, tpsmm_model_path, Ort::SessionOptions{ nullptr });
    this->_session_kp_detector = new Ort::Session(this->_env, kp_detector_model_path, Ort::SessionOptions{ nullptr });
}


void TPSMM::forward(const std::vector<float>& kp_source, const cv::Mat& source, const cv::Mat& driving, cv::Mat& generated){

    if(this->_session== nullptr){
        return;
    }
    std::vector<float> kp_source_value;
    kp_source_value.assign(kp_source.begin(), kp_source.end());
    // create input tensor object from data values

    Ort::Value kp_source_tensor = Ort::Value::CreateTensor<float>(this->_memory_info,
                                                                  kp_source_value.data(),
                                                                  this->_kp_size,
                                                                  this->_kp_dims.data(),
                                                                  this->_kp_dims.size());

    std::vector<float> source_value;
    source_value.assign(source.begin<float>(), source.end<float>());
    Ort::Value source_tensor = Ort::Value::CreateTensor<float>(this->_memory_info,
                                                               source_value.data(),
                                                               this->_input_size,
                                                               this->_input_dims.data(),
                                                               this->_input_dims.size());

    std::vector<float> driving_value;
    driving_value.assign(driving.begin<float>(), driving.end<float>());

    Ort::Value driving_tensor = Ort::Value::CreateTensor<float>(this->_memory_info,
                                                                driving_value.data(),
                                                                this->_input_size,
                                                                this->_input_dims.data(),
                                                                this->_input_dims.size());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(kp_source_tensor));
    ort_inputs.push_back(std::move(source_tensor));
    ort_inputs.push_back(std::move(driving_tensor));

    std::vector<float> outputTensorValues(this->_input_size);

    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            this->_memory_info, outputTensorValues.data(), this->_input_size,
            this->_input_dims.data(), this->_input_dims.size()));
    this->_session->Run(Ort::RunOptions{nullptr},
                        this->_input_node_names.data(),
                        ort_inputs.data(),
                        ort_inputs.size(),
                        this->_output_node_names.data(),
                        outputTensors.data(),
                        outputTensors.size());
    int siz[] = {1, 3, 256, 256};
    cv::Mat blob(4, siz, CV_32F);
    memcpy(blob.data, outputTensorValues.data(), outputTensorValues.size() * sizeof(float));

//    cv::dnn::imagesFromBlob(blob, generated);
    imagesFromBlob(blob, generated);

}


void TPSMM::kp_detector(const cv::Mat& input, std::vector<float>& output){
    if(this->_session_kp_detector== nullptr){
        return;
    }

    std::vector<float> source_value;
    source_value.assign(input.begin<float>(),
                        input.end<float>());

    // create input tensor object from data values
    auto source_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value source_tensor = Ort::Value::CreateTensor<float>(source_memory_info,
                                                               source_value.data(),
                                                               this->_input_size,
                                                               this->_input_dims.data(), 4);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(source_tensor));


    std::vector<float> outputTensorValues(this->_kp_size);
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            source_memory_info, outputTensorValues.data(), this->_kp_size,
            this->_kp_dims.data(), this->_kp_dims.size()));

    // score model & input tensor, get back output tensor
    this->_session_kp_detector->Run(Ort::RunOptions{nullptr},
                                    this->_kp_input_node_names.data(),
                                    inputTensors.data(),
                                    inputTensors.size(),
                                    this->_kp_output_node_names.data(),
                                    outputTensors.data(),
                                    outputTensors.size());

    // Get pointer to output tensor float values
    output.assign(outputTensorValues.begin(), outputTensorValues.end());
}