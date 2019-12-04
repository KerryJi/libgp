// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"

#include "cg.h"
#include "rprop.h"
#include <Eigen/Dense>

#include <fstream>
using namespace libgp;

const int feature_dim = 17;
const int feature_start = 0;
const int feature_end = 15;
const int feature_used = 16;
const int y_dim = 6;

const int train_index = 3;

void normalize(std::vector<double> &feature)
{
  if (feature.size() < feature_used)
    return;
  double sum = 0;
  for (int i = 0; i < feature_used; i++)
  {
    sum += feature[i];
  }
  if (sum < 0.000001)
    return;
  for (int i = 0; i < feature_used; i++)
  {
    feature[i] /= sum;
  }
}

void readFeature(std::string path, std::vector<std::vector<double>> &features)
{
  std::ifstream fs_(path);
  std::vector<double> feature(feature_used + y_dim);

  double temp;
  int input_i = 0;
  int need_i = 0;
  while (fs_ >> temp)
  {
    if (input_i >= feature_start && input_i <= feature_end)
    {
      feature[need_i++] = temp;
    }
    else if (input_i >= feature_dim)
    {
      feature[need_i++] = temp;
    }
    input_i++;
    if (need_i >= feature_used + y_dim)
    {
      // normalize(feature);
      features.push_back(feature);
      input_i = 0;
      need_i = 0;
    }
  }
}

int main(int argc, char const *argv[])
{
  std::string featurepath;
  std::string model_path = "model.txt";
  if (argc == 2)
  {
    featurepath = argv[1];
  }
  else if (argc == 3)
  {
    featurepath = argv[1];
    model_path = argv[2];
  }

  std::vector<std::vector<double>> features;
  readFeature(featurepath, features);
  std::cout << features.at(3).at(feature_used + train_index) << std::endl;
  // initialize Gaussian process for 2-D input using the squared exponential
  // covariance function with additive white noise.
  GaussianProcess gp(model_path.c_str());

  // total squared error
  double tss = 0;
  std::ofstream fs_output("output.txt");
  Eigen::VectorXd gt(features.size());
  Eigen::VectorXd estimate(features.size());
  gt.setZero();
  estimate.setZero();
  for (int i = 2; i < features.size(); ++i)
  {

    double f = gp.f(features.at(i).data());
    double error = f - features.at(i).at(feature_used + train_index);
    estimate(i) = f;
    gt(i) = features.at(i).at(feature_used + train_index);

    double var = std::sqrt(gp.var(features.at(i).data()));
    fs_output << f << " " << features.at(i).at(feature_used + train_index) << " " << error
              << " " << var << std::endl;
    tss += error * error;
    // tss += std::fabs(error);
  }
  estimate.normalize();
  gt.normalize();
  std::cout << "estimate dot gt:" << estimate.transpose() * gt << std::endl;
  std::cout << "mse = " << std::sqrt(tss / (features.size() - 2)) << std::endl;
  return EXIT_SUCCESS;
}
