#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse = VectorXd(5);

	int gt_size = ground_truth.size();
	int est_size = estimations.size();

	if(fabs(est_size)<0.001 || (gt_size - est_size) > 0) {
		cout<<"Error: size is not valid"<<endl;
		return rmse;
	}

	for(int idx=0; idx<gt_size; idx++) {
		VectorXd residual = estimations[idx] - ground_truth[idx];
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	rmse = (rmse/gt_size).array().sqrt();
	
	return rmse;
}