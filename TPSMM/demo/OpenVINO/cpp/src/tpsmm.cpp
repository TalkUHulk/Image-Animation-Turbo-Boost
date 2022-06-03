//
// Created by TalkUHulk on 2022/6/2.
//
#include "tpsmm.hpp"

static void blobFromImage(const cv::Mat& img, InferenceEngine::Blob::Ptr& blob){
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                           << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    float *blob_data = mblobHolder.as<float *>();

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob_data[c * img_w * img_h + h * img_w + w] =
                        (float)img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}


void TPSMM::init(const std::string& modelPath, const std::string& binPath,
                 const std::string& kp_modelPath, const std::string& kp_binPath,
                 const std::string& deviceName){
    // init kp
    this->_network_kp = this->_ie_core_kp.ReadNetwork(kp_modelPath, kp_binPath);
    InferenceEngine::InputInfo::Ptr input_info_kp = this->_network_kp.getInputsInfo().begin()->second;

    input_info_kp->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    input_info_kp->setLayout(InferenceEngine::Layout::NCHW);
    input_info_kp->setPrecision(InferenceEngine::Precision::FP32);

    InferenceEngine::DataPtr output_info_kp = this->_network_kp.getOutputsInfo().begin()->second;
    output_info_kp->setPrecision(InferenceEngine::Precision::FP32);

    this->_executable_network_kp = this->_ie_core_kp.LoadNetwork(this->_network_kp, deviceName);
    this->_infer_request_kp = this->_executable_network_kp.CreateInferRequest();

    //init tpsmm
    this->_network = this->_ie_core.ReadNetwork(modelPath, binPath);
    InferenceEngine::InputInfo::Ptr input_info = this->_network.getInputsInfo().begin()->second;

    input_info->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    input_info->setLayout(InferenceEngine::Layout::NCHW);
    input_info->setPrecision(InferenceEngine::Precision::FP32);

    InferenceEngine::DataPtr output_info = this->_network.getOutputsInfo().begin()->second;
    output_info->setPrecision(InferenceEngine::Precision::FP32);

    this->_executable_network = this->_ie_core.LoadNetwork(this->_network, deviceName);
    this->_infer_request = this->_executable_network.CreateInferRequest();
}

void TPSMM::kp_detector(const cv::Mat& input, InferenceEngine::Blob::Ptr &output){
    InferenceEngine::TensorDesc tDesc(
            InferenceEngine::Precision::FP32,
            { 1, (size_t)input.channels(), (size_t)input.size().height, (size_t)input.size().width },
            InferenceEngine::Layout::NHWC
    );
    InferenceEngine::Blob::Ptr imgBlob = InferenceEngine::make_shared_blob<float>(tDesc, (float*)input.data);

    auto input_name = this->_network_kp.getInputsInfo().begin()->first;
    this->_infer_request_kp.SetBlob(input_name, imgBlob);
    this->_infer_request_kp.Infer(); //sync

    auto output_name = this->_network_kp.getOutputsInfo().begin()->first;
    output = this->_infer_request_kp.GetBlob(output_name);

    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
    auto moutputHolder = moutput->rmap();
    float* net_pred = moutputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
}

//void TPSMM::forward(const InferenceEngine::Blob::Ptr &kp_source, const cv::Mat& source, const cv::Mat& driving, cv::Mat& output){
//    InferenceEngine::TensorDesc sourceDesc(
//            InferenceEngine::Precision::FP32,
//            { 1, (size_t)source.channels(), (size_t)source.size().height, (size_t)source.size().width },
//            InferenceEngine::Layout::NHWC
//    );
//    InferenceEngine::Blob::Ptr sourceBlob = InferenceEngine::make_shared_blob<float>(sourceDesc, (float*)source.data);
//    std::cout << "sourceBlob size:" << sourceBlob->size() << std::endl;
//
//    InferenceEngine::TensorDesc drivingDesc(
//            InferenceEngine::Precision::FP32,
//            { 1, (size_t)driving.channels(), (size_t)driving.size().height, (size_t)driving.size().width },
//            InferenceEngine::Layout::NHWC
//    );
//    InferenceEngine::Blob::Ptr drivingBlob = InferenceEngine::make_shared_blob<float>(drivingDesc, (float*)driving.data);
//
////    auto input_name = this->_network.getInputsInfo().begin()->first;
////    for(auto iter=this->_network.getInputsInfo().begin();iter != this->_network.getInputsInfo().end();++iter)
////        std::cout << "@@@ input_name:" << iter->first << std::endl;
//
//    this->_infer_request.SetBlob("driving", drivingBlob);
////    std::cout << "input_name:" << input_name << std::endl;
////
////
////    std::cout << "input_name:" << input_name << std::endl;
////
//    this->_infer_request.SetBlob("kp_source", kp_source);
//
//    std::cout << "kp_source size:" << kp_source->size() << std::endl;
//    std::cout << "drivingBlob size:" << drivingBlob->size() << std::endl;
//    std::cout << "sourceBlob size:" << sourceBlob->size() << std::endl;
//
////    this->_infer_request.SetBlob("source", sourceBlob);
////    std::cout << "input_name:" << input_name << std::endl;
//    InferenceEngine::Blob::Ptr imgBlob1 = this->_infer_request.GetBlob("driving");
//    InferenceEngine::Blob::Ptr imgBlob2 = this->_infer_request.GetBlob("source");
//    InferenceEngine::Blob::Ptr imgBlob3 = this->_infer_request.GetBlob("kp_source");
//    std::cout << "imgBlob1 size:" << imgBlob1->size() << std::endl;
//    std::cout << "imgBlob2 size:" << imgBlob2->size() << std::endl;
//    std::cout << "imgBlob3 size:" << imgBlob3->size() << std::endl;
////
////    imgBlob1 = drivingBlob;
////    imgBlob2 = sourceBlob;
////    imgBlob3 = kp_source;
//    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob2);
//    auto mblobHolder = mblob->wmap();
//    float *blob_data = mblobHolder.as<float *>();
//    for (size_t c = 0; c < 3; c++)
//    {
//        for (size_t  h = 0; h < 256; h++)
//        {
//            for (size_t w = 0; w < 256; w++)
//            {
//                blob_data[c * 256 * 256 + h * 256 + w] =
//                        (float)source.at<cv::Vec3b>(h, w)[c];
//            }
//        }
//    }
//
//
////    InferenceEngine::BlobMap inputs;
////    inputs.insert(std::make_pair("driving", drivingBlob));
////    inputs.insert(std::make_pair("source", sourceBlob));
////    inputs.insert(std::make_pair("kp_source", kp_source));
////    std::cout << "insert" << std::endl;
////    this->_infer_request.SetInput(inputs);
////    std::cout << "SetInput" << std::endl;
//
//    this->_infer_request.Infer(); //sync
//    std::cout << "Infer" << std::endl;
//    // get output
//    auto output_name = this->_network.getOutputsInfo().begin()->first;
//    InferenceEngine::Blob::Ptr output_blob = this->_infer_request.GetBlob(output_name);
//    std::cout << "output size:" << output_blob->size() << std::endl;
//
//    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
//    auto moutputHolder = moutput->rmap();
//    float* net_pred = moutputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
//    output = cv::Mat(256, 256, CV_32FC3);
//    for (int row = 0; row < 256; row++)
//    {
//        for (int col = 0; col < 256; col++)
//        {
//            //直接给第几行第几列赋值
//            output.at<cv::Vec3f>(row,col)[0] = net_pred[row * 256 + col];
//            output.at<cv::Vec3f>(row,col)[1] = net_pred[row * 256 + col + 256 * 256];
//            output.at<cv::Vec3f>(row,col)[2] = net_pred[row * 256 + col + 256 * 256 * 2];
//        }
//    }
//
//    std::cout << "output Mat:" << output.size() << std::endl;
//    std::ofstream outFile;
//    outFile.open("image.txt");
//    for(int i = 0; i < 196608; i++)
//    {
//        outFile << net_pred[i] <<",";
//    }
//    //关闭文件
//    outFile.close();
//}


void TPSMM::forward(const InferenceEngine::Blob::Ptr &kp_source, const cv::Mat& source, const cv::Mat& driving, cv::Mat& output){


    this->_infer_request.SetBlob("kp_source", kp_source);
    InferenceEngine::Blob::Ptr imgBlob1 = this->_infer_request.GetBlob("driving");
    InferenceEngine::Blob::Ptr imgBlob2 = this->_infer_request.GetBlob("source");
    InferenceEngine::Blob::Ptr imgBlob3 = this->_infer_request.GetBlob("kp_source");

    blobFromImage(source, imgBlob2);
    blobFromImage(driving, imgBlob1);

    this->_infer_request.Infer(); //sync
    // get output
    auto output_name = this->_network.getOutputsInfo().begin()->first;
    InferenceEngine::Blob::Ptr output_blob = this->_infer_request.GetBlob(output_name);

    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    auto moutputHolder = moutput->rmap();
    float* net_pred = moutputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    output = cv::Mat(256, 256, CV_32FC3);
    for (int row = 0; row < 256; row++)
    {
        for (int col = 0; col < 256; col++)
        {
            output.at<cv::Vec3f>(row,col)[0] = net_pred[row * 256 + col];
            output.at<cv::Vec3f>(row,col)[1] = net_pred[row * 256 + col + 256 * 256];
            output.at<cv::Vec3f>(row,col)[2] = net_pred[row * 256 + col + 256 * 256 * 2];
        }
    }

}

