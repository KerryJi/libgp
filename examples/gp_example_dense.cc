// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"

#include "cg.h"
#include "rprop.h"
#include <Eigen/Dense>

using namespace libgp;

int main(int argc, char const *argv[]) {
  int n = 1000, m = 1000;
  double tss = 0, error, f, y;
  // initialize Gaussian process for 2-D input using the squared exponential
  // covariance function with additive white noise.
  GaussianProcess gp(2, "CovSum ( CovRQiso, CovNoise)");
  // initialize hyper parameter vector
  Eigen::VectorXd params(gp.covf().get_param_dim());
  params << 2.0, 2.0, 2.0, -2.0;
  // set parameters of covariance function
  gp.covf().set_loghyper(params);

  // add training patterns
  for (int i = 0; i < n; ++i) {
    double x[] = {drand48() * 4 - 2, drand48() * 4 - 2};
    y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.5;
    gp.add_pattern(x, y);
  }
  libgp::RProp rprop;
  rprop.init();
  rprop.maximize(&gp, 50, true);
  // gp.log_likelihood();
  gp.write("model.txt");
  // total squared error
  for (int i = 0; i < m; ++i) {
    double x[] = {drand48() * 4 - 2, drand48() * 4 - 2};
    f = gp.f(x);
    y = Utils::hill(x[0], x[1]);
    error = f - y;
    tss += error * error;
  }
  std::cout << "mse = " << tss / m << std::endl;
  return EXIT_SUCCESS;
}
