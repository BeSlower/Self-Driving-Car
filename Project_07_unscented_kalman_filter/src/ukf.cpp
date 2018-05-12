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
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;
  
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

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sigma_points_ = 2 * n_aug_ + 1;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_points_);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(n_sigma_points_);
  weights_.segment(1, 2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Initialize Normalized Innovation Squared (NIS) value for both sensors
  NIS_laser_ = 0.;
  NIS_radar_ = 0.;

  // Measurement covariance matrices
  R_lidar_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);

  is_initialized_ = false;
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
  if(!is_initialized_) {

      // initialization
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Extract values from measurement
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rho_dot = meas_package.raw_measurements_(2);

      // Convert from polar to cartesian coordinates
      float px = rho * cos(phi);
      float py = rho * sin(phi);

      // Initialize state
      x_ << px, py, rho_dot, 0.0, 0.0;

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Extract values from measurement
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_(1);
      
      // Initialize state
      x_ << px, py, 0.0, 0.0, 0.0;
    }
    // Initialize state covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);

    // Update last measurement
    time_us_ = meas_package.timestamp_;

    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  

  // prediction
  Prediction(delta_t);

  // update
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // radar updates
    UpdateRadar(meas_package);
  } else {
    // lidar updates
    UpdateLidar(meas_package);
  }
}

MatrixXd UKF::PredictSigmaPoints(double dt) {

  // Augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // Compute sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
  Xsig_aug.col(0) = x_aug;
  MatrixXd L = P_aug.llt().matrixL();
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  for (int i = 0; i < n_sigma_points_; i++) {

    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001) {
      p_x = 0.1;
      p_y = 0.1;
    }

    // Predicted state values
    double px_p, py_p;
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * dt));
    }
    else {
      px_p = p_x + v * dt * cos(yaw);
      py_p = p_y + v * dt * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * dt;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * dt * dt * cos(yaw);
    py_p = py_p + 0.5 * nu_a * dt * dt * sin(yaw);
    v_p = v_p + nu_a * dt;
    yaw_p = yaw_p + 0.5 * nu_yawdd * dt * dt;
    yawd_p = yawd_p + nu_yawdd * dt;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  return Xsig_pred_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // -----------------------------------------
  //        generate and predict augment sigma points
  // -----------------------------------------
  MatrixXd Xsig_pred = PredictSigmaPoints(delta_t);

  // ----------------------------------------------
  //        predicted mean and covariance matrix
  // ----------------------------------------------
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  for (int i = 0; i < n_sigma_points_; i++) 
    x = x + weights_(i) * Xsig_pred.col(i);
  
  // Predicted state covariance matrix
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  for (int i = 0; i < n_sigma_points_; i++) {
    VectorXd x_diff = Xsig_pred.col(i) - x;
    // Normalize angle
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  // Update state vector and covariance matrix
  x_ = x;
  P_ = P;
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

  int n_z = 2;

  // Project sigma points onto measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sigma_points_); 

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
    VectorXd z_diff = Zsig.col(idx) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(idx) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  R_lidar_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;
  S = S + R_lidar_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int idx = 0; idx < 2 * n_aug_ + 1; idx++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(idx) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(idx) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(idx) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS for laser sensor
  NIS_laser_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
    (meas_package.raw_measurements_ - z_pred);

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

  int n_z = 3;
  
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

    if(fabs(py) > 0.001 && fabs(px) > 0.001)
      Zsig.col(idx)(1) = atan2(py, px);
    else
      Zsig.col(idx)(1) = 0.0;

    if(fabs(px*px + py*py) > 0.001)
      Zsig.col(idx)(2) = (px*v1 + py*v2) / (sqrt(px*px + py*py));
    else
      Zsig.col(idx)(2) = 0.0;
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
    VectorXd z_diff = Zsig.col(idx) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(idx) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  R_radar_ <<    std_radr_*std_radr_, 0, 0,
                  0, std_radphi_*std_radphi_, 0,
                  0, 0,std_radrd_*std_radrd_;
  S = S + R_radar_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int idx = 0; idx < 2 * n_aug_ + 1; idx++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(idx) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(idx) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(idx) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS for radar sensor
  NIS_radar_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
    (meas_package.raw_measurements_ - z_pred);
}