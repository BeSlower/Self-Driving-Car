#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // // -----------------------------------------
  // //        generate sigma points
  // // -----------------------------------------

  // MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // // calculate square root of P
  // MatrixXd A = P.llt().matrixL();

  // MatrixXd sig_pts_pos = MatrixXd(n_x_, n_x_);
  // MatrixXd sig_pts_neg = MatrixXd(n_x_, n_x_);
  // MatrixXd sqrt_lambda_P = sqrt(lambda_ + n_x_) * A;

  // for(int idx=0; idx<n_x; idx++) {
  //   sig_pts_pos.col(idx) = x_ + sqrt_lambda_P.col(idx);
  //   sig_pts_neg.col(idx) = x_ - sqrt_lambda_P.col(idx);
  // }

  // for(int idx=0; idx<n_x; idx++) {
  //   Xsig.col(idx+1) = sig_pts_pos.col(idx);
  //   Xsig.col(n_x_+idx+1) = sig_pts_neg.col(idx);
  // }

  // Xsig.col(0) = x_;

  // -----------------------------------------
  //        generate augment sigma points
  // -----------------------------------------
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // augmented state vector
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // augmented state covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_;

  // calculate square root of P_aug
  MatrixXd A = p.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for(int idx=0; idx<n_aug_; idx++) {
    Xsig_aug.col(idx+1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(idx);
    Xsig_aug.col(n_aug_+idx+1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(idx); 
  }

  // -----------------------------------------
  //        predict sigma points
  // -----------------------------------------
  VectorXd x_k = VectorXd(n_x_);
  VectorXd state_update = VectorXd(n_x_);
  VectorXd noise = VectorXd(n_x_);

  MatrixXd X_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for(int idx=0; idx < (2*n_aug_+1); idx++) {

    x_k = Xsig_aug.col(idx).head(n_x_);

    float px = Xsig_aug.col(idx)(0);
    float py = Xsig_aug.col(idx)(1);
    float v = Xsig_aug.col(idx)(2);
    float phi = Xsig_aug.col(idx)(3);
    float phi_dot = Xsig_aug.col(idx)(4);
    float v_a = Xsig_aug.col(idx)(5);
    float v_phi = Xsig_aug.col(idx)(6);

    // avoid division by zero
    if(phi_dot) {
      state_update(0) = v/phi * (sin(phi + phi*delta_t) - sin(phi));
      state_update(1) = v/phi_dot * (-cos(phi + phi*delta_t) + cos(phi));
    } else {
      state_update(0) = v * cos(phi) * delta_t;
      state_update(1) = v * sin(phi) * delta_t;
    }

    state_update(2) = 0;
    state_update(3) = phi_dot * delta_t;
    state_update(4) = 0;

    // calculate noise vector
    noise(0) = 0.5 * delta_t * delta_t * cos(phi) * v_a;
    noise(1) = 0.5 * delta_t * delta_t * sin(phi) * v_a;
    noise(2) = delta_t * v_a;
    noise(3) = 0.5 * delta_t * delta_t * v_phi;
    noise(4) = delta_t * v_phi;

    // write predicted sigma points into right column
    X_pred.col(idx) = x_k + state_update + noise;
  }

  // ----------------------------------------------
  //        predicted mean and covariance matrix
  // ----------------------------------------------

  // set weights
  VectorXd weights_ = VectorXd(2 * n_aug_ + 1);

  weights(0) = lambda_ / (lambda_+ n_aug_);
  for (int i=1; i<2*n_aug+1; i++) {
    double weight = 0.5/(n_aug+lambda);
    weights_(i) = weight;
  }

  // predicted mean
  VectorXd mean_pred = VectorXd(n_x_);
  mean_pred.fill(0.0);

  for(int idx=0; idx < (2*n_aug_+1); idx++) {
    mean_pred += weights_(idx) * X_pred.col(idx);
  }

  // predicted covariance matrix
  MatrixXd cov_pred = MatrixXd(n_x_, n_x_);
  cov_pred.fill(0.0);

  VectorXd diff = VectorXd(n_x_);
  for(int idx = 0; idx < 2 * n_aug_ + 1; idx ++) {

    // state difference
    VectorXd x_diff = X_pred.col(i) - mean_pred;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    cov_pred = cov_pred + weights_(i) * x_diff * x_diff.transpose() ;
  }

  Xsig_pred_ = X_pred;
  x_ = mean_pred;
  P_ = cov_pred;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd z = meas_package.raw_measurements_();

  MatrixXd H = MatrixXd(2, 5);
  MatrixXd R = MatrixXd(2, 2);

  H << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0;
  R << std_laspx_*std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  VectorXd y = z - H * x_;
  MatrixXd S = H * P_ * H.transpose() + R;
  MatrixXd K = P_ * H * S.inverse();

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  x_ = x_ + K * y;
  P_ = (I - K * H) * P_;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  n_z = 3;
  
  // predicted sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  for(int idx = 0; idx < 2*n_aug_+1; idx++) {
    double px = Xsig_pred_.col(idx)(0);
    double py = Xsig_pred_.col(idx)(1);
    double v = Xsig_pred_.col(idx)(2);
    double yaw = Xsig_pred_.col(idx)(3);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig.col(idx)(0) = sqrt(px*px + py*py);
    Zsig.col(idx)(1) = atan2(py, px);
    Zsig.col(idx)(2) = (px*v1 + py*v2) / (sqrt(px*px + py*py));
  }

  // predicted mean in measurement space
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  for(int idx = 0; idx < 2*n_aug_+1; idx++) {
    z_pred = z_pred + weights_(idx) * Zsig.col(idx);
  }

  // predicted covariance matrix in measurement space
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  for (int idx = 0; idx < 2 * n_aug_ + 1; idx++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int idx = 0; idx < 2 * n_aug_ + 1; idx++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  z = meas_package.raw_measurements_();
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}
