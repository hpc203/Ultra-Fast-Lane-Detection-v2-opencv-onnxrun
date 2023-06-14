#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

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
	Mat normalize_(Mat img);
	float SoftMaxFast(const float* src, float* dst, int64_t length);
	int inpWidth;
	int inpHeight;
	float mean[3] = { 0.485, 0.456, 0.406 };
	float std[3] = { 0.229, 0.224, 0.225 };
	string dataset;
	int num_row;
	int num_col;
	vector<float> row_anchor;
	vector<float> col_anchor;
	float crop_ratio;
	void GenerateAnchor();

	Net net;
};

Ultra_Fast_Lane_Detection_v2::Ultra_Fast_Lane_Detection_v2(string model_path)
{
	this->net = readNet(model_path);

	size_t pos = model_path.find("_res");
	int len = pos - 15;
	dataset = model_path.substr(15, len);
	if (dataset == "culane")
	{
		num_row = 72;
		num_col = 81;
		crop_ratio = 0.6;
		inpHeight = 320;
		inpWidth = 1600;
	}
	else
	{
		num_row = 56;
		num_col = 41;
		crop_ratio = 0.8;
		inpHeight = 320;
		inpWidth = 800;
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
	row_anchor.clear();
	col_anchor.clear();
}

Mat Ultra_Fast_Lane_Detection_v2::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	vector<cv::Mat> bgrChannels(3);
	split(img, bgrChannels);
	for (int c = 0; c < 3; c++)
	{
		bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1.0 / (255.0* std [c]), (0.0 - mean[c]) / std[c]);
	}
	Mat m_normalized_mat;
	merge(bgrChannels, m_normalized_mat);
	Rect rect(0, row - inpHeight, col, inpHeight);
	Mat dstimg = m_normalized_mat(rect);
	return dstimg;
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
	Mat img;
	resize(srcimg, img, Size(this->inpWidth, int(float(this->inpHeight) / this->crop_ratio)), INTER_LINEAR);
	Mat dstimg = this->normalize_(img);
	Mat blob = blobFromImage(dstimg);

	net.enableWinograd(false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理

	////pred2coords
	const float* loc_row = (float*)outs[3].data;
	const float* loc_col = (float*)outs[2].data;
	const float* exist_row = (float*)outs[1].data;
	const float* exist_col = (float*)outs[0].data;
	
	vector<int64_t> loc_row_dims = { outs[3].size[0],outs[3].size[1],outs[3].size[2], outs[3].size[3] };
	int64_t num_grid_row = loc_row_dims[1];
	int64_t num_cls_row = loc_row_dims[2];
	int64_t num_lane_row = loc_row_dims[3];
	vector<int64_t> loc_col_dims = { outs[2].size[0],outs[2].size[1],outs[2].size[2], outs[2].size[3] };
	int64_t num_grid_col = loc_col_dims[1];
	int64_t num_cls_col = loc_col_dims[2];
	int64_t num_lane_col = loc_col_dims[3];

	vector<int64_t> exist_row_dims = { outs[1].size[0],outs[1].size[1],outs[1].size[2], outs[1].size[3] };
	vector<int64_t> exist_col_dims = { outs[0].size[0],outs[0].size[1],outs[0].size[2], outs[0].size[3] };

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

	static const string kWinName = "Deep learning lane detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, drawimg);
	waitKey(0);
	destroyAllWindows();
}
