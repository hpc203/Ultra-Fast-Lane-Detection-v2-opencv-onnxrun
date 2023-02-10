#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

vector<int64_t> argmax_1(const float* v, const vector<int64_t>& dims)
{
	vector<int64_t> ret;
	ret.resize(dims[0] * dims[2] * dims[3]);

	for (int64_t i = 0; i < dims[2]; i++) {
		for (int64_t j = 0; j < dims[3]; j++) {
			int64_t offset = dims[3] * i + j;
			float max_val = 0;
			int64_t max_index = 0;
			for (int64_t k = 0; k < dims[1]; k++) {
				size_t index = k * dims[2] * dims[3] + offset;
				if (v[index] > max_val) {
					max_val = v[index];
					max_index = k;
				}
			}
			ret[offset] = max_index;
		}
	}
	return ret;
}

int64_t sum_valid(const vector<int64_t>& v, int64_t num, int64_t interval, int64_t offset)
{
	int64_t sum = 0;
	for (int64_t i = 0; i < num; i++) {
		sum += v[i * interval + offset];
	}
	return sum;
}

inline float fast_exp(float x)
{
	union {
		uint64_t i;
		float f;
	} v{};
	v.i = static_cast<int64_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
	return v.f;
}

class Ultra_Fast_Lane_Detection_v2
{
public:
	Ultra_Fast_Lane_Detection_v2(string model_path);
	Mat detect(Mat& cv_image);
	~Ultra_Fast_Lane_Detection_v2();  // 析构函数, 释放内存

private:
	void normalize_(Mat img);
	float SoftMaxFast(const float* src, float* dst, int64_t length);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	float mean[3] = { 0.485, 0.456, 0.406 };
	float std[3] = { 0.229, 0.224, 0.225 };
	string dataset;
	int num_row;
	int num_col;
	vector<float> row_anchor;
	vector<float> col_anchor;
	float crop_ratio;
	void GenerateAnchor();

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Ultra-Fast-Lane-Detection-v2");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

Ultra_Fast_Lane_Detection_v2::Ultra_Fast_Lane_Detection_v2(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

	size_t pos = model_path.find("_res");
	int len = pos - 15;
	dataset = model_path.substr(15, len);
	if (dataset == "culane")
	{
		num_row = 72;
		num_col = 81;
		crop_ratio = 0.6;
	}
	else
	{
		num_row = 56;
		num_col = 41;
		crop_ratio = 0.8;
	}
	GenerateAnchor();
}

void Ultra_Fast_Lane_Detection_v2::GenerateAnchor()
{
	for (int i = 0; i < num_row; i++)
	{
		if (dataset == "culane")
		{
			row_anchor.push_back(0.42 + i * (1.0 - 0.42) / (num_row - 1));
		}
		else
		{
			row_anchor.push_back((160 + i * (710 - 160) / (num_row - 1)) / 720.0);
		}
	}
	for (int i = 0; i < num_col; i++)
	{
		col_anchor.push_back(0.0 + i * (1.0 - 0.0) / (num_col - 1));
	}
}

Ultra_Fast_Lane_Detection_v2::~Ultra_Fast_Lane_Detection_v2()
{
	input_image_.clear();
	row_anchor.clear();
	col_anchor.clear();
}

void Ultra_Fast_Lane_Detection_v2::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(inpHeight * inpWidth * img.channels());
	int ind = 0;
	for (int c = 0; c < 3; c++)
	{
		for (int i = row - inpHeight; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[ind++] = (pix / 255.0 - mean[c]) / std[c];
			}
		}
	}
}

float Ultra_Fast_Lane_Detection_v2::SoftMaxFast(const float* src, float* dst, int64_t length)
{
	const float alpha = *std::max_element(src, src + length);
	float denominator{ 0 };

	for (int64_t i = 0; i < length; ++i) {
		dst[i] = fast_exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int64_t i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

Mat Ultra_Fast_Lane_Detection_v2::detect(Mat& srcimg)
{
	const int img_h = srcimg.rows;
	const int img_w = srcimg.cols;
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, int(float(this->inpHeight) / this->crop_ratio)), INTER_LINEAR);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理

	////pred2coords
	const float* loc_row = ort_outputs[0].GetTensorMutableData<float>();
	const float* loc_col = ort_outputs[1].GetTensorMutableData<float>();
	const float* exist_row = ort_outputs[2].GetTensorMutableData<float>();
	const float* exist_col = ort_outputs[3].GetTensorMutableData<float>();
	
	vector<int64_t> loc_row_dims = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64_t num_grid_row = loc_row_dims[1];
	int64_t num_cls_row = loc_row_dims[2];
	int64_t num_lane_row = loc_row_dims[3];
	vector<int64_t> loc_col_dims = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	int64_t num_grid_col = loc_col_dims[1];
	int64_t num_cls_col = loc_col_dims[2];
	int64_t num_lane_col = loc_col_dims[3];

	vector<int64_t> exist_row_dims = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	vector<int64_t> exist_col_dims = ort_outputs[3].GetTensorTypeAndShapeInfo().GetShape();

	vector<int64_t> max_indices_row = argmax_1(loc_row, loc_row_dims);
	vector<int64_t> valid_row = argmax_1(exist_row, exist_row_dims);
	vector<int64_t> max_indices_col = argmax_1(loc_col, loc_col_dims);
	vector<int64_t> valid_col = argmax_1(exist_col, exist_col_dims);

	vector<vector<Point>> line_list(4);
	for (int64_t i : { 1, 2 })
	{
		if (sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row * 0.5)
		{
			for (int64_t k = 0; k < num_cls_row; k++)
			{
				int64_t index = k * num_lane_row + i;
				if (valid_row[index] != 0)
				{
					vector<float> pred_all_list;
					vector<int64_t> all_ind_list;
					for (int64_t all_ind = max(0, int(max_indices_row[index] - 1)); all_ind <= (min(num_grid_row - 1, max_indices_row[index]) + 1); all_ind++)
					{
						pred_all_list.push_back(loc_row[all_ind * num_cls_row * num_lane_row + index]);
						all_ind_list.push_back(all_ind);
					}
					vector<float> pred_all_list_softmax(pred_all_list.size());
					this->SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
					float out_temp = 0;
					for (int64_t l = 0; l < pred_all_list.size(); l++)
					{
						out_temp += pred_all_list_softmax[l] * all_ind_list[l];
					}
					float x = (out_temp + 0.5) / (num_grid_row - 1.0);
					float y = row_anchor[k];
					line_list[i].push_back(Point(int(x*img_w), int(y*img_h)));
				}
			}
		}
	}

	for (int64_t i : {0, 3})
	{
		if (sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 4)
		{
			for (int64_t k = 0; k < num_cls_col; k++)
			{
				int64_t index = k * num_lane_col + i;
				if (valid_col[index] != 0)
				{
					vector<float> pred_all_list;
					vector<int64_t> all_ind_list;
					for (int64_t all_ind = max(0, int(max_indices_col[index] - 1)); all_ind <= (min(num_grid_col - 1, max_indices_col[index]) + 1); all_ind++)
					{
						pred_all_list.push_back(loc_col[all_ind * num_cls_col * num_lane_col + index]);
						all_ind_list.push_back(all_ind);
					}
					vector<float> pred_all_list_softmax(pred_all_list.size());
					this->SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
					float out_temp = 0;
					for (int64_t l = 0; l < pred_all_list.size(); l++)
					{
						out_temp += pred_all_list_softmax[l] * all_ind_list[l];
					}
					float y = (out_temp + 0.5) / (num_grid_col - 1.0);
					float x = col_anchor[k];
					line_list[i].push_back(Point(int(x*img_w), int(y*img_h)));
				}
			}
		}
	}
	Mat drawimg = srcimg.clone();
	for (auto& line : line_list)
	{
		for (auto& p : line)
		{
			circle(drawimg, p, 3, Scalar(0, 255, 0), -1);
		}

	}

	return drawimg;
}

int main()
{
	Ultra_Fast_Lane_Detection_v2 mynet("weights/ufldv2_culane_res18_320x1600.onnx");
	string imgpath = "images/culane/00000.jpg";
	Mat srcimg = imread(imgpath);
	Mat drawimg = mynet.detect(srcimg);

	static const string kWinName = "Deep learning lane detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, drawimg);
	waitKey(0);
	destroyAllWindows();
}