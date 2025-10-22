# Bags

There are two types of bag files in this repo. The first are recordings from the Olin College Miller Academic Center (MAC) made using a physical robot. The second type is derived from those recordings, specifically `macfirst_floor_take_1` and the `maps/mac_1st_floor_final.yaml` map. These derived files are designed to demonstrate the particle filter’s localization capabilities.

Both of the derived bag files can be played in rviz2 using the `rviz/amcl.rviz` configuration. Among others, they have the key topics `/map` to display the map, `/particle_cloud` to display the particles, `/tf` to broadcast the current estimated transform from robot to map frames, `/robot_pose_estimate` which displays the current estimate of the location, and `/scan` which enables the user to compare the current robot's scan (again derived from `macfirst_floor_take_1`) against the map. The more closely they align, the better our estimated tranformation between the robot and map frames are.

The bag `full_map_init` shows the particle filter converge when the particle's are initialzed randomly across the map. The bag `pose_estimate_init` shows the particle converge rapidly when given an initial pose estimate. 


