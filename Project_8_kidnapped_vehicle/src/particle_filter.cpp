/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
// #include <limits>

using namespace std;

#include "particle_filter.h"


// Create only once the default random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// TODO: Set standard deviations for x, y, and theta
	num_particles = 200;

 	normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		
		// TODO: Sample  and from these normal distrubtions like this: 
		//	 sample_x = dist_x(gen);
		//	 where "gen" is the random engine initialized earlier.
		
		// initialize particles
		Particle p;	 
 		p.id = i;
 		p.x = dist_x(gen);
 		p.y = dist_y(gen);
 		p.theta = dist_theta(gen);
 		p.weight = 1.0;

 		particles.push_back(p);
 		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double x, y, theta, x_f, y_f, theta_f;

	for(int i = 0; i < num_particles; i++) {
		
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;

		if(abs(yaw_rate) > 1e-5) {
			theta_f = theta + yaw_rate * delta_t;
			x_f = x + velocity / yaw_rate * (sin(theta_f) - sin(theta));
			y_f = y + velocity / yaw_rate * (cos(theta) - cos(theta_f));
		} else {
			theta_f = theta;
			x_f = x + velocity * delta_t * cos(theta);
			y_f = y + velocity * delta_t * sin(theta);
		}

		normal_distribution<double> dist_x(x_f, std_pos[0]);
		normal_distribution<double> dist_y(y_f, std_pos[1]);
		normal_distribution<double> dist_theta(theta_f, std_pos[2]);
		
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (unsigned int i = 0; i < observations.size(); i++) {
        double min_dist = std::numeric_limits<double>::max();
        int min_index = -1;

        for (unsigned int j = 0; j < predicted.size(); j++) {
            double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_index = j;
            }
        }
        // assign nearest neighbor
        observations[i].id = predicted[min_index].id;
        // play some trick here, use delta distance instead of real distance
        observations[i].x -= predicted[min_index].x;
        observations[i].y -= predicted[min_index].y;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double x, y, theta;

	// landmark stand deviation
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];



	for(int i = 0; i < num_particles; i++) {

		// current particle values
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;

		vector<LandmarkObs> landmark_in_map;
		vector<LandmarkObs> obs_in_map;

		// find possible landmarks in range
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			int land_id = map_landmarks.landmark_list[j].id_i;
			double land_x = map_landmarks.landmark_list[j].x_f;
			double land_y = map_landmarks.landmark_list[j].y_f;

			double distance = dist(x, y, land_x, land_y);

			if(distance <= sensor_range) {
				LandmarkObs land_pred;
                land_pred.id = land_id;
                land_pred.x = land_x;
                land_pred.y = land_y;
                landmark_in_map.push_back(land_pred);
			}
		}

		// all observation and convert observation to map's coordinate system
		for(unsigned int j = 0; j < observations.size(); j++) {
			LandmarkObs land_obs;
			land_obs.id = -1;
			land_obs.x = x + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
			land_obs.y = y + sin(theta) * observations[j].x + cos(theta) * observations[j].y;
			obs_in_map.push_back(land_obs);
		}

		// filter observations by the nearest neighbor selection (association)
		dataAssociation(landmark_in_map, obs_in_map);

		// update weight by calculating multivariate gaussian probability
		double weight = 1.0;
		for(unsigned int idx = 0; idx < obs_in_map.size(); idx++) {
			double diff_x = obs_in_map[idx].x;
			double diff_y = obs_in_map[idx].y;

			weight *= 1.0/(2.0*M_PI*std_x*std_y) * exp(-0.5 * (diff_x * diff_x / (std_x*std_x) + diff_y * diff_y / (std_y*std_y)));
		}

		weights[i] = weight;
	}

	//normalize weights
	double norm_factor = 0.0;
	for(int i = 0; i < num_particles; i++) 
		norm_factor += weights[i];

	for(int i = 0; i < num_particles; i++) {
		weights[i] /= norm_factor;
		particles[i].weight = weights[i];
	}
 }

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int> weighted_distribution(weights.begin(), weights.end());

    vector<Particle> resampled_particles;
    for (int i = 0; i < num_particles; ++i) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
