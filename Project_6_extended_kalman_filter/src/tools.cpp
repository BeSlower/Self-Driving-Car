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

	for(int idx=0; idx < est_size; idx++) {
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
	
	// Pre-compute some term which recur in the Jacobian
	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = c1 * c2;

	if (std::abs(c1) < 0.0001) {
		std::cout << "Error in CalculateJacobian. Division by zero." << std::endl;
		return Hj;
	}

	// compute Jacobian matrix
	Hj << (px / c2),				(py / c2),					0,			0,
		-(py / c1),					(px / c1),					0,			0,
		py * (vx*py - vy*px) / c3,	px * (vy*px - vx*py) / c3,	px / c2,	py / c2;

	return Hj;
}