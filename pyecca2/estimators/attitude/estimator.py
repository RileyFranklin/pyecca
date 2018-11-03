import time

import numpy as np

import pyecca2.msgs as msgs
import pyecca2.uros as uros


class AttitudeEstimator:
    """
    An attitude estimator node for uros
    """

    def __init__(self, core, name, eqs):
        self.core = core
        self.name = name

        # subscriptions
        self.sub_imu = uros.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.sub_mag = uros.Subscriber(core, 'mag', msgs.Mag, self.mag_callback)

        # publications
        self.pub_est = uros.Publisher(core, name + '_status', msgs.EstimatorStatus)
        self.pub_state = uros.Publisher(core, name + '_state', msgs.VehicleState)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_state = msgs.VehicleState()

        self.sub_params = uros.Subscriber(core, 'params', msgs.Params, self.params_callback)

        # parameters
        self.param_list = []

        def add_param(name, value, type):
            p = uros.Param(self.core, self.name + '/' + name, value, type)
            self.param_list.append(p)
            return p

        self.std_mag = add_param('std_mag', 1e-3, 'f8')
        self.std_accel = add_param('std_accel', 1e-3, 'f8')
        self.std_accel_omega = add_param('std_accel_omega', 1e-6, 'f8')

        self.std_gyro = add_param('std_gyro', 1e-6, 'f8')
        self.sn_gyro_rw = add_param('sn_gyro_rw', 1e-6, 'f8')
        self.mag_decl = add_param('mag_decl', 0, 'f8')
        self.beta_mag_c = add_param('beta_mag_c', 100, 'f8')
        self.beta_accel_c = add_param('beta_accel_c', 100, 'f8')
        self.g = add_param('g', 9.8, 'f8')

        # misc
        self.x = eqs['constants']()['x0']
        self.W = eqs['constants']()['W0']
        self.n_x = self.x.shape[0]
        self.n_e = self.W.shape[0]
        self.t_last_imu = 0
        self.eqs = eqs

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def mag_callback(self, msg):
        y = msg.data['mag']
        # out: ['x_mag', 'W_mag', 'beta_mag', 'r_mag', 'r_std_mag', 'error_code'])
        # in: ['x', 'W', 'y_b', 'decl', 'std_mag', 'beta_mag_c'],
        self.x, self.W, beta_mag, r_mag, r_std_mag, mag_ret = self.eqs['correct_mag'](
            self.x, self.W, y, self.mag_decl.get(), self.std_mag.get(), self.beta_mag_c.get())
        self.msg_est_status.data['beta_mag'] = beta_mag
        self.msg_est_status.data['r_mag'][:r_mag.shape[0]] = r_mag
        self.msg_est_status.data['r_std_mag'][:r_std_mag.shape[0]] = r_std_mag
        self.msg_est_status.data['mag_ret'] = mag_ret

    def imu_callback(self, msg):

        # compute dt
        t = msg.data['time']
        dt = t - self.t_last_imu
        self.t_last_imu = t

        # estimate state
        omega = msg.data['gyro']
        start = time.thread_time()
        if dt > 0:
            self.x, self.W = self.eqs['predict'](t, self.x, self.W, omega,
                                                 self.std_mag.get(), self.sn_gyro_rw.get(), dt)
        q, r, b_g = self.eqs['get_state'](self.x)
        end = time.thread_time()
        elapsed = end - start

        # correct accel
        # out: ['x_accel', 'W_accel', 'beta_accel', 'r_accel', 'r_std_accel', 'error_code'])
        # in: ['x', 'W', 'y_b', 'g', 'omega_b', 'std_accel', 'std_accel_omega', 'beta_accel_c'],
        self.x, self.W, beta_accel, r_accel, r_std_accel, accel_ret = self.eqs['correct_accel'](
            self.x, self.W, msg.data['accel'], self.g.get(), omega,
            self.std_accel.get(), self.std_accel_omega.get(), self.beta_accel_c.get())
        self.msg_est_status.data['beta_accel'] = beta_accel
        self.msg_est_status.data['r_accel'][:r_accel.shape[0]] = r_accel.T
        self.msg_est_status.data['r_std_accel'][:r_accel.shape[0]] = r_std_accel.T
        self.msg_est_status.data['accel_ret'] = accel_ret

        # publish vehicle state
        self.msg_state.data['time'] = t
        self.msg_state.data['q'] = q.T
        self.msg_state.data['r'] = r.T
        self.msg_state.data['b'] = b_g.T
        self.msg_state.data['omega'] = omega.T
        self.pub_state.publish(self.msg_state)

        # publish estimator status
        self.msg_est_status.data['time'] = t
        self.msg_est_status.data['n_x'] = self.n_x
        self.msg_est_status.data['x'][:self.n_x] = self.x.T
        W_vect = np.reshape(np.array(self.W)[np.diag_indices(self.n_e)], -1)
        self.msg_est_status.data['W'][:len(W_vect)] = W_vect
        self.msg_est_status.data['elapsed'] = elapsed
        self.pub_est.publish(self.msg_est_status)
