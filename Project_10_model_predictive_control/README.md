# Model Predictive Control (MPC)
This project implenments a Model Predictive Controller for keeping the car on the track in the simulator.

### Overview

![mpc](mpc.gif)

**Model predictive control (MPC)** is an advanced method of process control which relies on dynamic models of the process. Differently from previously implemented PID controller, MPC controller has the ability to anticipate future events and can take control actions accordingly. Indeed, future time steps are taking into account while optimizing current time slot.

The MPC controller framework consists in four main components:

- **Trajectory** taken in consideration during optimization. This is parametrized by a number of time steps **\*N*** spaced out by a time **\*dt***. 

- **Vehicle Model**, which is the set of equations that describes system behavior and updates across time steps. In our case, we used a simplified kinematic model (so called *bycicle model*) described by a state of six parameters:

  |  State   |    Definition     |
  | :------: | :---------------: |
  |  **x**   |    x position     |
  |  **y**   |    y position     |
  | **psi**  | heading direction |
  |  **v**   |     velocity      |
  | **cte**  | cross-track error |
  | **epsi** | orientation error |

- **Contraints** necessary to model contrants in actuators' respose. For instance, a vehicle will never be able to steer 90 deegrees in a single time step. In this project we set these constraints as follows:

  - **steering**: bounded in range [-25°, 25°]
  - **acceleration**: bounded in range [-1, 1] from full brake to full throttle

- **Cost Function** is the objective function we are trying to optimize. This function is based on the motion model and control process. It is the sum of several terms. Besides the main terms that depends on reference values (*e.g.* cross-track error, orientation error), other regularization terms are present to enforce the smoothness in the controller response (*e.g.* avoid abrupt steering).

### Tuning Trajectory Parameters

Both N and dt are fundamental parameters in the optimization process. In particular, T = N \* dt constitutes the *prediction horizon* considered during optimization. These values have to be tuned keeping in mind a couple of things:

- large *dt* result in less frequent actuations, which in turn could result in the difficulty in following a continuous reference trajectory (so called *discretization error*)
- despite the fact that having a large *T* could benefit the control process, consider that predicting too far in the future does not make sense in real-world scenarios.
- large *T* and small *dt* lead to large *N*. As mentioned above, the number of variables optimized is directly proportional to *N*, so will lead to an higher computational cost.

In the current project I empirically set (by visually inspecting the vehicle's behavior in the simulator) these parameters to be N=25 and dt=0.04, for a total of T=1s in the future.

### Changing Reference System

Simulator provides coordinates in global reference system. In order to ease later computation, these are converted into car's own reference system at lines 98-105 in main.cpp.

### Dealing with Latency

To model real driving conditions where the car does actuate the commands instantly, a *100ms* latency delay has been introduced before sending the data message to the simulator (line 185 in main.cpp). In order to deal with latency, state is predicted one time step ahead before feeding it to the solver (lines 124-132 in main.cpp).

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).
