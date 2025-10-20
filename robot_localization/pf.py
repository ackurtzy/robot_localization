#!/usr/bin/env python3

"""This is the starter code for the robot localization project"""

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
import random
from occupancy_field import OccupancyField
from helper_functions import TFHelper, draw_random_sample
from angle_helpers import quaternion_from_euler


class Particle(object):
    """
    Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)

    Attributes:
        x: the x-coordinate of the hypothesis relative to the map frame
        y: the y-coordinate of the hypothesis relative ot the map frame
        theta: the yaw of the hypothesis relative to the map frame
        w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """
        Construct a new Particle

        Arguments:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
        """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """
        A helper function to convert a particle to a geometry_msgs/Pose message
        """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(
            position=Point(x=self.x, y=self.y, z=0.0),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )


class ParticleFilter(Node):
    """The class that represents a Particle Filter ROS Node
    Attributes list:
        base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
        map_frame: the name of the map coordinate frame (should be "map" in most cases)
        odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
        scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
        n_particles: the number of particles in the filter
        d_thresh: the amount of linear movement before triggering a filter update
        a_thresh: the amount of angular movement before triggering a filter update
        pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
        particle_pub: a publisher for the particle cloud
        last_scan_timestamp: this is used to keep track of the clock when using bags
        scan_to_process: the scan that our run_loop should process next
        occupancy_field: this helper class allows you to query the map for distance to closest obstacle
        transform_helper: this helps with various transform operations (abstracting away the tf2 module)
        particle_cloud: a list of particles representing a probability distribution over robot poses
        current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                               The pose is expressed as a list [x,y,theta] (where theta is the yaw)
        thread: this thread runs your main loop
    """

    def __init__(self):
        """
        Initialize the particle filter
        """
        super().__init__("pf")
        self.base_frame = "base_footprint"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame
        self.scan_topic = "scan"  # the topic where we will get laser scans from

        self.n_particles = 300  # the number of particles to use

        self.d_thresh = 0.2  # the amount of linear movement before performing an update
        self.a_thresh = (
            math.pi / 10
        )  # the amount of angular movement before performing an update

        self.resampling_radius_scaling_factor = (
            0.5  # Size of radius in particle resampling
        )
        self.resampling_angle_scaling_factor = (
            1  # Standard deviation of noise in angle resampling sampling
        )

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(
            PoseWithCovarianceStamped, "initialpose", self.update_initial_pose, 10
        )

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", 10)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """
        Sends out the map to odom transform
        """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(
            seconds=0.1
        )
        self.transform_helper.send_last_map_to_odom_transform(
            self.map_frame, self.odom_frame, postdated_timestamp
        )

    def loop_wrapper(self):
        """
        Wraps the run_loop in a timer

        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2
        """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """
        The main run loop of the particle filter.

        Calls helper functions for each core processing operation.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(
            self.odom_frame, self.base_frame, msg.header.stamp
        )
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return

        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(
            msg, self.base_frame
        )
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            self.odom_pose
        )
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud()
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.update_particles_with_odom()  # update based on odometry
            # self.update_particles_with_laser(r, theta)  # update based on laser scan
            self.update_robot_pose()  # update robot's pose based on particles
            self.resample_particles()  # resample particles to focus on areas of high density
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        """
        Check if the robot has moved beyond an angle or distance threshold.

        Args:
            new_odom_xy_theta (tuple/list): An (x,y,yaw) tuple of the new
            robot's position
        """
        return (
            math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0])
            > self.d_thresh
            or math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1])
            > self.d_thresh
            or math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2])
            > self.a_thresh
        )

    def update_robot_pose(self):
        """
        Update the estimate of the robot's pose with the particles.

        Chooses the particle with the highest weight and updates
        the map to odom transform according to it's location.
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        highest_weight = -1
        best_particle = None
        for part in self.particle_cloud:
            if part.w > highest_weight:
                best_particle = part

        if hasattr(self, "odom_pose") and best_particle:
            self.robot_pose = best_particle.as_pose()
            self.transform_helper.fix_map_to_odom_transform(
                self.robot_pose, self.odom_pose
            )
        elif best_particle is None:
            self.get_logger().warn("Can't update robot pose. No particles")
        else:
            self.get_logger().warn(
                "Can't set map->odom transform since no odom data received"
            )

    def update_particles_with_odom(self):
        """
        Update the particles using the newly given odometry pose.

        To do this, it first computes change in x, y, and theta location in the
        odom frame.

        To compute the new particle location, it rotates the change in position
        vector so that the x value is the front/back change and y is the
        left/right relative to the old position. Next, it rotates this change to
        align with each particle's heading and adds this change to the particle's
        position.

        To compute the new particle angle, it adds the change in theta to each
        particle's theta value.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            self.odom_pose
        )
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            delta = (
                new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                new_odom_xy_theta[2] - self.current_odom_xy_theta[2],
            )

            original_theta = self.current_odom_xy_theta[2]
            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        r_mat_relative_xy = np.array(
            (
                [np.cos(-original_theta), -np.sin(-original_theta)],
                [np.sin(-original_theta), np.cos(-original_theta)],
            )
        )
        relative_xy = r_mat_relative_xy @ np.array(delta[:2]).reshape(-1, 1)

        print(
            f"Relative Movement, Original theta: {np.rad2deg(original_theta)}, Current theta: {np.rad2deg(new_odom_xy_theta[2])}\nForward: {relative_xy[0]}, Side: {relative_xy[1]}"
        )

        for part in self.particle_cloud:
            r_mat_relative_to_particle = np.array(
                (
                    [np.cos(part.theta), -np.sin(part.theta)],
                    [np.sin(part.theta), np.cos(part.theta)],
                )
            )
            old_position = np.array(([part.x], [part.y]))
            new_position = old_position + (r_mat_relative_to_particle @ relative_xy)

            part.x = float(new_position[0])
            part.y = float(new_position[1])
            part.theta = part.theta + delta[2]

    def resample_particles(self):
        """
        Resample the particle cloud based on current particle weights.

        First, it normalizes the particle weights to ensure they form a valid
        probability distribution. Then, it draws a weighted random sample of
        particles proportional to their weights.

        For each selected particle, it adds noise in both position and
        heading. The positional noise is sampled from a Gaussian distribution
        scaled inverse to the particle's weight, then converted to polar
        coordinates to determine the noise's direction and magnitude. The
        angular noise is similarly sampled and added to the particle's heading.
        """
        # make sure the distribution is normalized
        self.normalize_particles()

        base_particles = draw_random_sample(
            [part for part in self.particle_cloud],
            [part.w for part in self.particle_cloud],
            self.n_particles,
        )

        for index, b_part in enumerate(base_particles):
            radius = np.random.normal(
                0, self.resampling_radius_scaling_factor * (1 - b_part.w), 1
            )
            pol_angle = np.random.uniform(0, 2 * math.pi, 1)

            new_x = b_part.x + radius * np.cos(pol_angle)
            new_y = b_part.y + radius * np.sin(pol_angle)

            new_theta = b_part.theta + np.random.normal(
                0, self.resampling_angle_scaling_factor * (1 - b_part.w), 1
            )

            print(f"Resampled particle {index}")
            print(
                f"  From: ({self.particle_cloud[index].x}, {self.particle_cloud[index].y}, {np.rad2deg(self.particle_cloud[index].theta)})"
            )
            self.particle_cloud[index].x = float(new_x)
            self.particle_cloud[index].y = float(new_y)
            self.particle_cloud[index].theta = float(new_theta) % (2.0 * math.pi)
            self.particle_cloud[index].w = 1.0
            print(f"New coords: ({new_x}, {new_y})")
            print(
                f"  To: ({self.particle_cloud[index].x}, {self.particle_cloud[index].y}, {np.rad2deg(self.particle_cloud[index].theta)})"
            )

    def update_particles_with_laser(self, r, theta):
        """Updates the particle weights in response to the scan data
        r: the distance readings to obstacles
        theta: the angle relative to the robot frame for each corresponding reading

        Taking in the current laser scan data of the Neato, it removes large radius values
        then converts them from polar form (r, theta) to cartesian form (x, y).

        Iterating through each particle in the cloud, it notes the heading and rotates the laser
        scan endpoints by that amount, translating them by the particles coordinates. Using
        the occupancy field method to get the closest object to a coordinate and subtracting every
        value by the largest error, the particle weight are set.
        """

        # Removes radius values that are greater than a certain value
        for i in range(len(r)):
            if radius > 5:
                del r[i]
                del theta[i]

        # Creates an empty array for x,y coordinates for scan endpoints
        x_orig = np.array([])
        y_orig = np.array([])

        # Converts the scan endpoints from r,theta to x,y
        for i, radius in enumerate(r):
            x_orig[i] = radius * np.cos(theta[i])
            y_orig[i] = radius * np.sin(theta[i])

        # Creating new arrays for the updated values of x,y for each scan endpoint
        x_final = np.array([])
        y_final = np.array([])
        part_weights = np.array([])

        # Iterating through each particle, creating a rotation matrix using its heading, and rotating each of the scan endpoints
        for i, part in enumerate(self.particle_cloud):

            rotation_amt = part.theta

            rotation_matrix = np.array(
                (
                    [np.cos(rotation_amt), -np.sin(rotation_amt)],
                    [np.sin(rotation_amt), np.cos(rotation_amt)],
                )
            )

            # Applies the rotation matrix to the x,y of the particle using matrix multiplication
            current_x_y = np.array([x_orig[i], y_orig[i]]).reshape(-1, 1)
            rotated_x_y = rotation_matrix @ current_x_y

            # Translates the scan endpoint using the position of the particle
            x_final[i] = float(rotated_x_y[0]) + part.x
            y_final[i] = float(rotated_x_y[1]) + part.y

            # Fills the weight array with distance values
            part_weights[i] = OccupancyField.get_closest_obstacle_distance(x_final[i], y_final[i])

        # Finds the max distance
        max_distance = np.max(part_weights)
        
        # Subtracts every distance by the max distance 
        for i, part in enumerate(self.particle_cloud):
            part_weights[i] -= max_distance
            part.w = part_weights[i]

    def update_initial_pose(self, msg):
        """
        Callback function to handle re-initializing the particle filter based on a pose estimate.
        """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(xy_theta)

    def initialize_particle_cloud(self, xy_theta=None):
        """
        Initialize the particle cloud with a given pose estimate or random sampling.

        Args
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to
                initialize the particle cloud around.

        If no pose estimate is provided, it samples particle positions
        uniformly across the bounds of the occupancy field and gives each
        particle a random heading.

        If a pose estimate is given, it distributes particles around the estimate
        with Gaussian noise added to each particle's location. It gives each particle
        a random orientation.
        """
        self.particle_cloud = []
        if xy_theta is None:
            x_bounds, y_bounds = self.occupancy_field.get_obstacle_bounding_box()

            for _ in range(self.n_particles):
                init_x = random.random() * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
                init_y = random.random() * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
                init_theta = random.random() * 2 * math.pi

                self.particle_cloud.append(
                    Particle(x=init_x, y=init_y, theta=init_theta)
                )
        else:
            x_guess, y_guess, _ = xy_theta
            radius = np.random.normal(0, 2, self.n_particles)
            pol_angle = np.random.random(self.n_particles) * 2 * math.pi

            init_xs = x_guess + radius * np.cos(pol_angle)
            init_ys = y_guess + radius * np.sin(pol_angle)
            init_thetas = np.random.random(self.n_particles) * 2 * math.pi

            for i in range(self.n_particles):
                self.particle_cloud.append(
                    Particle(
                        x=float(init_xs[i]),
                        y=float(init_ys[i]),
                        theta=float(init_thetas[i]),
                    )
                )

        self.normalize_particles()

    def normalize_particles(self):
        """
        Make sure the particle weights define a valid distribution (i.e. sum to 1.0)
        """
        norm_factor = 0.0
        for part in self.particle_cloud:
            norm_factor += part.w

        if norm_factor - 0.0 < 0.00001:
            return

        for part in self.particle_cloud:
            part.w /= norm_factor

    def publish_particles(self, timestamp):
        """
        Publish the particle cloud to be visualized.

        Args:
            timestamp: The timestamp to give the particle cloud message.
        """
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        """
        Filter to wait to update scan internally until it the old scan is
        processed
        """
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop
        if self.scan_to_process is None:
            self.scan_to_process = msg


def main(args=None):
    """
    Intialize and run the particle filter
    """
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
