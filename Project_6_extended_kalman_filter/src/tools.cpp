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
	VectorXd rmse(4);
	rmse.fill(0);

	int est_size = estimations.size();
	int gt_size = ground_truth.size();

	if(fabs(est_size - gt_size) > 0.0 || est_size ==0) {
		cout<<"Error: estimations size error"<<endl;
		return rmse;
	}

	// calculate RMSE value

	for(int idx=0; idx<est_size; idx++) {
		VectorXd residual = estimations[idx] - ground_truth[idx];
		residual = residual.array() * residual.array();
		rmse += residual ;
	}

	rmse = (rmse / est_size).array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	MatrixXd Hj = MatrixXd(3, 4);
	Hj.fill(0.0);

	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float square = px*px + py*py;
	float sq_root = sqrt(square);

	if(fabs(square) < 0.0001) {
		cout<<"Error: divided by zero"<<endl;
		return Hj;
	}

	Hj << px/sq_root, py/sq_root, 0, 0,
		  -py/square, px/square, 0, 0,
		  py*(vx*py-vy*px)/(square*sq_root), px*(vy*px-vx*py)/(square*sq_root), px/sq_root, py/sq_root;

	return Hj;
}
