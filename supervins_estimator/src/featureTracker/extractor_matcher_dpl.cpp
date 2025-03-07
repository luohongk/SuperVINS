#include "extractor_matcher_dpl.h"

Extractor_DPL::Extractor_DPL(unsigned int _num_threads) : num_threads(_num_threads)
{
}

void Extractor_DPL::initialize(std::string extractorPath, int extractor_type_)
{
    extractor_type = extractor_type_;
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");
    session_options = Ort::SessionOptions();
    session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.gpu_mem_limit = 0;
    cuda_options.arena_extend_strategy = 1;     // 设置GPU内存管理中的Arena扩展策略
    cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
    cuda_options.has_user_compute_stream = 0;
    cuda_options.default_memory_arena_cfg = nullptr;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Session = std::make_unique<Ort::Session>(env, extractorPath.c_str(), session_options);

    // Initial Extractor
    size_t numInputNodes = Session->GetInputCount();
    InputNodeNames.reserve(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++)
    {
        InputNodeNames.emplace_back(strdup(Session->GetInputNameAllocated(i, allocator).get()));
        InputNodeShapes.emplace_back(Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    size_t numOutputNodes = Session->GetOutputCount();
    OutputNodeNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        OutputNodeNames.emplace_back(strdup(Session->GetOutputNameAllocated(i, allocator).get()));
        OutputNodeShapes.emplace_back(Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
}

cv::Mat Extractor_DPL::pre_process(const cv::Mat &Image, float &scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resize_img = ResizeImage(tempImage, IMAGE_SIZE_DPL, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(resize_img);
    if (extractor_type ==SUPERPOINT && tempImage.channels()==3)
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        resultImage = RGB2Grayscale(resultImage);
    }
    return resultImage;
}

std::pair<std::vector<cv::Point2f>, float *> Extractor_DPL::extract_featurepoints(const cv::Mat &image)
{
    int InputTensorSize;
    if (extractor_type  == SUPERPOINT)
    {
        InputNodeShapes[0] = {1, 1, image.size().height, image.size().width};
    }
    else if (extractor_type == DISK)
    {
        InputNodeShapes[0] = {1, 3, image.size().height, image.size().width};
    }

    InputTensorSize = InputNodeShapes[0][0] * InputNodeShapes[0][1] * InputNodeShapes[0][2] * InputNodeShapes[0][3];

    std::vector<float> srcInputTensorValues(InputTensorSize);

    if (extractor_type  == SUPERPOINT)
    {
        srcInputTensorValues.assign(image.begin<float>(), image.end<float>());
    }
    else
    {
        int height = image.rows;
        int width = image.cols;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Vec3f pixel = image.at<cv::Vec3f>(y, x); // RGB
                srcInputTensorValues[y * width + x] = pixel[2];
                srcInputTensorValues[height * width + y * width + x] = pixel[1];
                srcInputTensorValues[2 * height * width + y * width + x] = pixel[0];
            }
        }
    }

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                          OrtMemType::OrtMemTypeCPU);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(),
        InputNodeShapes[0].data(), InputNodeShapes[0].size()));

    auto output_tensor = Session->Run(Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
                                      input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

    for (auto &tensor : output_tensor)
    {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
        }
    }

    outputtensors.emplace_back(std::move(output_tensor));
    std::pair<std::vector<cv::Point2f>, float *> result_pts_descriptors = post_process(std::move(outputtensors[0]));

    outputtensors.clear();

    return result_pts_descriptors;
}

std::pair<std::vector<cv::Point2f>, float *> Extractor_DPL::post_process(std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f>, float *> extractor_result;
    std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t *kpts = (int64_t *)tensor[0].GetTensorMutableData<void>();

    std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
    float *scores = (float *)tensor[1].GetTensorMutableData<void>();

    std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
    float *desc = (float *)tensor[2].GetTensorMutableData<void>();
    std::vector<cv::Point2f> kpts_f;
    for (int i = 0; i < kpts_Shape[1] * 2; i += 2)
    {
        kpts_f.emplace_back(cv::Point2f(kpts[i], kpts[i + 1]));
    }

    extractor_result.first = kpts_f;
    extractor_result.second = desc;
    return extractor_result;
}

void Matcher_DPL::initialize(std::string matcherPath,int extractor_type_, float matchThresh_)
{
    matchThresh= matchThresh_;
    cout << "match threshold = "<<matchThresh<<endl;
    extractor_type = extractor_type_;

    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Matcher");
    session_options = Ort::SessionOptions();
    session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // use gpu
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.gpu_mem_limit = 0;
    cuda_options.arena_extend_strategy = 1;     // 设置GPU内存管理中的Arena扩展策略
    cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
    cuda_options.has_user_compute_stream = 0;
    cuda_options.default_memory_arena_cfg = nullptr;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Session = std::make_unique<Ort::Session>(env, matcherPath.c_str(), session_options);

    // Initial Extractor
    size_t numInputNodes = Session->GetInputCount();
    InputNodeNames.reserve(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++)
    {
        InputNodeNames.emplace_back(strdup(Session->GetInputNameAllocated(i, allocator).get()));
        InputNodeShapes.emplace_back(Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    size_t numOutputNodes = Session->GetOutputCount();
    OutputNodeNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        OutputNodeNames.emplace_back(strdup(Session->GetOutputNameAllocated(i, allocator).get()));
        OutputNodeShapes.emplace_back(Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
}

Matcher_DPL::Matcher_DPL(unsigned int _num_threads) : num_threads(_num_threads)
{
}

std::vector<cv::Point2f> Matcher_DPL::pre_process(std::vector<cv::Point2f> kpts, int h, int w)
{
    return NormalizeKeypoints(kpts, h, w);
}

std::vector<std::pair<int, int>> Matcher_DPL::match_featurepoints(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1)
{
    InputNodeShapes[0] = {1, static_cast<int>(kpts0.size()), 2};
    InputNodeShapes[1] = {1, static_cast<int>(kpts1.size()), 2};
    if (extractor_type  == SUPERPOINT)
    {
    InputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 256};
    InputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 256};
    }
    else
    {
        InputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 128};
        InputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 128};
    }

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    float *kpts0_data = new float[kpts0.size() * 2];
    float *kpts1_data = new float[kpts1.size() * 2];

    for (size_t i = 0; i < kpts0.size(); ++i)
    {
        kpts0_data[i * 2] = kpts0[i].x;
        kpts0_data[i * 2 + 1] = kpts0[i].y;
    }
    for (size_t i = 0; i < kpts1.size(); ++i)
    {
        kpts1_data[i * 2] = kpts1[i].x;
        kpts1_data[i * 2 + 1] = kpts1[i].y;
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, kpts0_data, kpts0.size() * 2 * sizeof(float),
        InputNodeShapes[0].data(), InputNodeShapes[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, kpts1_data, kpts1.size() * 2 * sizeof(float),
        InputNodeShapes[1].data(), InputNodeShapes[1].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc0, kpts0.size() * 256 * sizeof(float),
        InputNodeShapes[2].data(), InputNodeShapes[2].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc1, kpts1.size() * 256 * sizeof(float),
        InputNodeShapes[3].data(), InputNodeShapes[3].size()));

    auto output_tensor = Session->Run(Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
                                      input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

    for (auto &tensor : output_tensor)
    {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
            std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
        }
    }
    outputtensors = std::move(output_tensor);

    std::vector<std::pair<int, int>> result_matches = post_process();

    outputtensors.clear();

    return result_matches;
}

std::vector<std::pair<int, int>> Matcher_DPL::post_process()
{
    std::vector<std::pair<int, int>> good_matches;
    // load date from tensor
    std::vector<int64_t> matches_Shape = outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t *matches = (int64_t *)outputtensors[0].GetTensorMutableData<void>();
    std::vector<int64_t> mscore_Shape = outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
    float *mscores = (float *)outputtensors[1].GetTensorMutableData<void>();
    for (int i = 0; i < matches_Shape[0]; i++)
    {
        if (mscores[i] > this->matchThresh)
        {
            good_matches.emplace_back(std::make_pair(matches[i * 2], matches[i * 2+1]));
        }
    }
    return good_matches;
}