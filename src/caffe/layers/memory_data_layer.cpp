#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  n_labels_ = this->layer_param_.memory_data_param().n_labels();
  if (!n_labels_) {
    n_labels_ = 1;
  }
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  label_shape_.push_back(batch_size_);
  std::cout << "n_lables: " << n_labels_ << std::endl;
  label_shape_.push_back(n_labels_);
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape_);
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(label_shape_);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add.";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  label_shape_[0] = num;
  added_label_.Reshape(label_shape_);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = datum_vector[item_id].label();
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
    const vector<Dtype>& labels) {
  size_t num = mat_vector.size();
  CHECK(!has_new_data_) <<
      "Can't add mat until current data has been consumed.";
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  CHECK_EQ(num * n_labels_, labels.size()) <<
      "The labels size must be equal to the number of images multiplied "
              "by the number of labels per image";
  added_data_.Reshape(num, channels_, height_, width_);
  label_shape_[0] = num;
  added_label_.Reshape(label_shape_);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = labels[item_id];
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  label_shape_[0] = batch_size_;
  added_label_.Reshape(label_shape_);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::HandleGenerators() {
  if (generate_cv_mat_labels_cb_) {
    std::vector<cv::Mat> mats;
    std::vector<Dtype> labels;
    generate_cv_mat_labels_cb_->generate(batch_size_, &mats, &labels);
    AddMatVector(mats, labels);
  } else if (generate_datum_cb_) {
    std::vector<Datum> data;
    generate_datum_cb_->generate(batch_size_, &data);
    AddDatumVector(data);
  } else if (generate_raw_pointer_cb_) {
    Dtype * data;
    Dtype * labels;
    int n;
    generate_raw_pointer_cb_->generate(batch_size_, &data, &labels, &n);
    Reset(data, labels, n);
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if ((!data_ || !has_new_data_)) {
    HandleGenerators();
  }
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape_);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  top[1]->set_cpu_data(labels_ + pos_ * n_labels_);
  pos_ = (pos_ + batch_size_) % n_;
  if (pos_ == 0)
    has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe
