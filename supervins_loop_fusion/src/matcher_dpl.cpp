/**
 * @file matcher_dpl.cpp
 * @brief LightGlue feature matcher for loop closure verification
 * @note Adapted from supervins_estimator/src/featureTracker/extractor_matcher_dpl.cpp
 *       Only Matcher_DPL implementation (Extractor not needed in loop module)
 */

#include "matcher_dpl.h"

Matcher_DPL::Matcher_DPL(unsigned int _num_threads) : num_threads(_num_threads)
{
}

void Matcher_DPL::initialize(std::string matcherPath, int extractor_type_, float matchThresh_)
{
    matchThresh = matchThresh_;
    std::cout << "[LoopMatcher] match threshold = " << matchThresh << std::endl;
    extractor_type = extractor_type_;

    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LoopClosure_LightGlue_Matcher");
    session_options = Ort::SessionOptions();
    // GPU推理时CPU线程仅负责数据搬运和调度，无需大量线程
    // When using GPU inference, CPU threads only handle data transfer and scheduling
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // use GPU
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.gpu_mem_limit = 0;
    cuda_options.arena_extend_strategy = 1;
    cuda_options.do_copy_in_default_stream = 1;
    cuda_options.has_user_compute_stream = 0;
    cuda_options.default_memory_arena_cfg = nullptr;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Session = std::make_unique<Ort::Session>(env, matcherPath.c_str(), session_options);

    // Initialize Matcher
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

    std::cout << "[LoopMatcher] LightGlue model loaded successfully: " << matcherPath << std::endl;
}

std::vector<cv::Point2f> Matcher_DPL::pre_process(std::vector<cv::Point2f> kpts, int h, int w)
{
    return NormalizeKeypoints(kpts, h, w);
}

std::vector<std::pair<int, int>> Matcher_DPL::match_featurepoints(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1)
{
    // 记录原始大小，用于后续过滤填充点的匹配
    const int orig_size0 = kpts0.size();
    const int orig_size1 = kpts1.size();

    // RTX 40系列 GPU 在特征点数量较少时会触发 packed QKV bug
    // 通过填充到最小尺寸 256 来规避此问题
    const int MIN_KPT_SIZE = 256;
    int desc_dim = (extractor_type == LOOP_SUPERPOINT) ? 256 : 128;

    // 填充 kpts0 和 desc0
    std::vector<float> padded_desc0;
    if (orig_size0 < MIN_KPT_SIZE)
    {
        padded_desc0.resize(MIN_KPT_SIZE * desc_dim, 0.0f);
        memcpy(padded_desc0.data(), desc0, orig_size0 * desc_dim * sizeof(float));
        desc0 = padded_desc0.data();
        for (int i = orig_size0; i < MIN_KPT_SIZE; i++)
            kpts0.push_back(cv::Point2f(-10.0f, -10.0f)); // 填充无效坐标
    }

    // 填充 kpts1 和 desc1
    std::vector<float> padded_desc1;
    if (orig_size1 < MIN_KPT_SIZE)
    {
        padded_desc1.resize(MIN_KPT_SIZE * desc_dim, 0.0f);
        memcpy(padded_desc1.data(), desc1, orig_size1 * desc_dim * sizeof(float));
        desc1 = padded_desc1.data();
        for (int i = orig_size1; i < MIN_KPT_SIZE; i++)
            kpts1.push_back(cv::Point2f(-10.0f, -10.0f)); // 填充无效坐标
    }

    InputNodeShapes[0] = {1, static_cast<int64_t>(kpts0.size()), 2};
    InputNodeShapes[1] = {1, static_cast<int64_t>(kpts1.size()), 2};
    if (extractor_type == LOOP_SUPERPOINT)
    {
        InputNodeShapes[2] = {1, static_cast<int64_t>(kpts0.size()), 256};
        InputNodeShapes[3] = {1, static_cast<int64_t>(kpts1.size()), 256};
    }
    else
    {
        InputNodeShapes[2] = {1, static_cast<int64_t>(kpts0.size()), 128};
        InputNodeShapes[3] = {1, static_cast<int64_t>(kpts1.size()), 128};
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
        memory_info_handler, kpts0_data, kpts0.size() * 2,
        InputNodeShapes[0].data(), InputNodeShapes[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, kpts1_data, kpts1.size() * 2,
        InputNodeShapes[1].data(), InputNodeShapes[1].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc0, kpts0.size() * desc_dim,
        InputNodeShapes[2].data(), InputNodeShapes[2].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc1, kpts1.size() * desc_dim,
        InputNodeShapes[3].data(), InputNodeShapes[3].size()));

    std::vector<Ort::Value> output_tensor;
    try
    {
        output_tensor = Session->Run(Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
                                     input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());
    }
    catch (const Ort::Exception &e)
    {
        // GPU 推理失败（如 packed QKV 不支持特定输入大小），返回空匹配
        std::cerr << "[WARN][LoopMatcher] GPU inference failed: " << e.what() << std::endl;
        std::cerr << "[WARN][LoopMatcher] Skipping matching (kpts0=" << orig_size0 << ", kpts1=" << orig_size1 << ")" << std::endl;
        delete[] kpts0_data;
        delete[] kpts1_data;
        return std::vector<std::pair<int, int>>();
    }

    for (auto &tensor : output_tensor)
    {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
            std::cerr << "[ERROR][LoopMatcher] Inference output tensor is not a tensor or doesn't have value" << std::endl;
        }
    }
    outputtensors = std::move(output_tensor);

    std::vector<std::pair<int, int>> result_matches = post_process();

    outputtensors.clear();

    // 释放 kpts 数据
    delete[] kpts0_data;
    delete[] kpts1_data;

    // 过滤掉涉及填充点的匹配（索引 >= 原始大小的为填充点）
    if (orig_size0 < MIN_KPT_SIZE || orig_size1 < MIN_KPT_SIZE)
    {
        std::vector<std::pair<int, int>> filtered_matches;
        for (auto &m : result_matches)
        {
            if (m.first < orig_size0 && m.second < orig_size1)
            {
                filtered_matches.push_back(m);
            }
        }
        return filtered_matches;
    }

    return result_matches;
}

std::vector<std::pair<int, int>> Matcher_DPL::post_process()
{
    std::vector<std::pair<int, int>> good_matches;
    // load data from tensor
    std::vector<int64_t> matches_Shape = outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t *matches = (int64_t *)outputtensors[0].GetTensorMutableData<void>();
    std::vector<int64_t> mscore_Shape = outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
    float *mscores = (float *)outputtensors[1].GetTensorMutableData<void>();
    float score_sum = 0.0f;
    for (int i = 0; i < matches_Shape[0]; i++)
    {
        if (mscores[i] > this->matchThresh)
        {
            good_matches.emplace_back(std::make_pair(matches[i * 2], matches[i * 2 + 1]));
            score_sum += mscores[i];
        }
    }
    // 计算平均匹配置信度
    last_avg_match_score = good_matches.empty() ? 0.0f : score_sum / good_matches.size();
    return good_matches;
}
