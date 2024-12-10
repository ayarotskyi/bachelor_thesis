import numpy as np
import pygame
import math
import os
from tqdm import tqdm
import csv

saved_image_index = 0
class VehicleTrajectorySimulator:
    def __init__(self, joystick_data, image_directory, screen_width=800, screen_height=800):
        """
        Initialize the trajectory simulator and visualization.
        
        :param joystick_data: Numpy array with dtype [('x', '<f8'), ('y', '<f8'), ('timestamp', '<M8[ms]')]
        :param image_directory: Directory containing images to be displayed
        :param screen_width: Width of the visualization window
        :param screen_height: Height of the visualization window
        """
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Vehicle Trajectory Simulation with Images")
        self.clock = pygame.time.Clock()
        
        # Simulation area parameters
        self.AREA_WIDTH = 2.0   # 2 meters wide
        self.AREA_HEIGHT = 2.0  # 1 meter tall
        
        # Screen dimensions
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        
        # Image display area
        self.IMAGE_HEIGHT = screen_height // 2
        self.IMAGE_WIDTH = screen_width

        # Load images
        self.images = self.load_images(image_directory)
        
        # Vehicle and simulation parameters
        self.wheel_base = 0.17  # meters
        self.motor_efficiency = 0.67
        self.speed_threashold = 0.30
        
        # Initial vehicle position (30cm from one long side)
        self.initial_x = 0
        self.initial_y = 0.6
        
        # Simulation state
        self.current_x = self.initial_x
        self.current_y = self.initial_y
        self.current_theta = 0
        
        # Trajectory data
        self.joystick_data = joystick_data
        self.trajectory = self.simulate_trajectory()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)

        self.current_image_index = 0
    
    def load_images(self, directory):
        """
        Load images from the specified directory.
        
        :param directory: Path to directory containing images
        :return: List of pygame surfaces
        """
        # Get all image files, sorted to ensure correct order
        def key(file_name):
            return int(file_name[:-4])
        image_files = sorted(os.listdir(directory), key=key)
        
        # Load images
        images = []
        for filename in image_files:
            try:
                # Full path to the image
                full_path = os.path.join(directory, filename)
                
                # Load image
                image = pygame.image.load(full_path)
                
                # Resize image to fit the designated area
                image = pygame.transform.scale(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
                
                images.append(image)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        
        return images
    
    def calculate_motor_speeds(self, x, y):
        rotation_quotient = 0.5
        # Calculate left and right motor powers
        left_power = -y + x * rotation_quotient
        right_power = -y - x * rotation_quotient

        # Normalize powers to ensure they're within -1 to 1 range
        max_power = max(abs(left_power), abs(right_power), 1)
        left_power /= max_power
        right_power /= max_power

        return left_power, right_power
    
    def simulate_trajectory(self):
        """
        Simulate vehicle trajectory with acceleration and previous speed consideration.

        :return: List of trajectory points with position and timestamp
        """
        trajectory = []

        # Initial conditions
        current_x, current_y = self.initial_x, self.initial_y
        current_theta = 0

        # Initial velocities
        current_linear_velocity = 0
        current_angular_velocity = 0

        # Vehicle parameters for more realistic dynamics
        MAX_LINEAR_ACCELERATION = 1.0  # m/s²
        MAX_ANGULAR_ACCELERATION = 1.0  # rad/s²
        MAX_LINEAR_VELOCITY = 2.0  # m/s
        MAX_ANGULAR_VELOCITY = math.pi  # rad/s

        for i, point in enumerate(self.joystick_data):
            # Calculate motor speeds
            left_speed, right_speed = self.calculate_motor_speeds(point['x'], point['y'])

            # Convert motor speeds to desired velocities
            desired_left_motor_velocity = (left_speed * self.motor_efficiency 
                                           if left_speed > self.speed_threashold
                                           or left_speed < -self.speed_threashold
                                           else 0)
            desired_right_motor_velocity = (right_speed * self.motor_efficiency 
                                            if right_speed > self.speed_threashold
                                            or right_speed < -self.speed_threashold
                                            else 0)

            # Calculate desired linear and angular velocities
            desired_linear_velocity = (desired_left_motor_velocity + desired_right_motor_velocity) / 2
            desired_angular_velocity = (desired_right_motor_velocity - desired_left_motor_velocity) / self.wheel_base

            # Calculate acceleration (limited by max acceleration)
            linear_acceleration = np.clip(
                desired_linear_velocity - current_linear_velocity, 
                -MAX_LINEAR_ACCELERATION, 
                MAX_LINEAR_ACCELERATION
            )
            angular_acceleration = np.clip(
                desired_angular_velocity - current_angular_velocity, 
                -MAX_ANGULAR_ACCELERATION, 
                MAX_ANGULAR_ACCELERATION
            )

            # Update velocities with acceleration
            current_linear_velocity = np.clip(
                current_linear_velocity + linear_acceleration, 
                -MAX_LINEAR_VELOCITY, 
                MAX_LINEAR_VELOCITY
            )
            current_angular_velocity = np.clip(
                current_angular_velocity + angular_acceleration, 
                -MAX_ANGULAR_VELOCITY, 
                MAX_ANGULAR_VELOCITY
            )

            # Time between points (in seconds)
            if i > 0:
                time_delta = (point['timestamp'] - self.joystick_data[i-1]['timestamp']).astype('float64') / 1000.0
            else:
                time_delta = 0.0

            # Update position and orientation
            current_theta += current_angular_velocity * time_delta
            current_x += current_linear_velocity * np.cos(current_theta) * time_delta
            current_y += current_linear_velocity * np.sin(current_theta) * time_delta

            # Store trajectory point
            trajectory.append({
                'x': current_x,
                'y': current_y,
                'timestamp': point['timestamp']
            })
        return trajectory
    
    def hyperparameter(self):
        print("starting hyperparameter search")
        desired_x = 0.0
        desired_y = 0.8
        desired_max_x = 1.8
        best_point_x = 100.0
        best_point_y = 100.0
        best_diff = 100.0
        best_motor_efficiency = 0.0
        best_wheel_base = 0.0
        best_speed_threashold = 0.0
        for wheel_base in tqdm([wheel_base / 1000 for wheel_base in range(10, 500, 10)]):
            for motor_efficiency in tqdm([motor_efficiency / 100 for motor_efficiency in range(1, 100, 1)], leave=False):
                for speed_threashold in tqdm([speed_threashold / 100 for speed_threashold in range(0, 60, 1)], leave=False):
                    self.wheel_base = wheel_base
                    self.motor_efficiency = motor_efficiency
                    self.speed_threashold = speed_threashold
                    trajectory = self.simulate_trajectory()
                    last_point = trajectory[-1]
                    diff = (math.sqrt(math.pow(last_point['x'] - desired_x, 2) + math.pow(last_point['y'] - desired_y, 2)) 
                            + abs(max(map(lambda point: point['x'], trajectory)) - desired_max_x))
                    if (diff < best_diff):
                        best_point_x = last_point['x']
                        best_point_y = last_point['y']
                        best_diff = diff
                        best_motor_efficiency = self.motor_efficiency
                        best_wheel_base = self.wheel_base
                        best_speed_threashold = self.speed_threashold
        print("best parameters:")
        print("wheel base: ", best_wheel_base)
        print("motor efficiency: ", best_motor_efficiency)
        print("speed threashold: ", best_speed_threashold)
        print("end point: x:", best_point_x, "y:", best_point_y)
        print("max diff:", best_diff)

    def save_reduced_data(self):
        global saved_image_index
        if not os.path.exists("D:/bachelor arbeit/reduced_data/images"):
            os.makedirs("D:/bachelor arbeit/reduced_data/images")
        for image in self.images[:self.current_image_index]:
            pygame.image.save(image, "D:/bachelor arbeit/reduced_data/images/"+str(saved_image_index)+".png")
            saved_image_index = saved_image_index + 1
        self.append_numpy_array_to_csv(self.joystick_data[:self.current_image_index], "D:/bachelor arbeit/reduced_data/data.csv")
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def append_numpy_array_to_csv(self, data, filename):
        try:
            start_time = data[0]['timestamp']
            # Open the file in append mode
            with open(filename, 'a', newline='') as csvfile:
                # Create a CSV writer
                csv_writer = csv.writer(csvfile)

                # Check if the file is empty, if so, write the header
                csvfile.seek(0, 2)  # Move to the end of the file
                if csvfile.tell() == 0:
                    csv_writer.writerow(['x', 'y', 'ms_from_start'])

                # Convert each row to a format suitable for CSV writing
                for row in data:
                    diff_ms = (row['timestamp'] - start_time).astype('timedelta64[ms]')
                    ms_from_start_str = str(diff_ms.astype('int'))

                    # Write the row to the CSV file
                    csv_writer.writerow([
                        row['x'], 
                        row['y'], 
                        ms_from_start_str
                    ])

            print(f"Successfully appended {len(data)} rows to {filename}")

        except IOError as e:
            print(f"Error writing to CSV file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def convert_to_screen_coords(self, x, y):
        """
        Convert simulation coordinates to screen coordinates.
        
        :param x: X coordinate in simulation space
        :param y: Y coordinate in simulation space
        :return: Tuple of (screen_x, screen_y)
        """
        screen_x = int((x / self.AREA_WIDTH) * (self.SCREEN_WIDTH))
        screen_y = int(((self.AREA_HEIGHT - y) / self.AREA_HEIGHT) * (self.SCREEN_HEIGHT // 2)) - self.SCREEN_HEIGHT / 2
        return screen_x, screen_y
    
    def run_simulation(self):
        """
        Run the trajectory simulation and visualization.
        """
        previous_frame_time = pygame.time.get_ticks()
        running = True
        simulation_speed = 0.0  # Adjustable simulation speed
        elapsed_time = 0.0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        simulation_speed *= 1.5
                    elif event.key == pygame.K_MINUS:
                        simulation_speed /= 1.5
                    elif event.key == pygame.K_SPACE:
                        if simulation_speed == 0:
                            simulation_speed = 1.0
                        else:
                            simulation_speed = 0.0
                    elif event.key == pygame.K_RIGHT:
                        elapsed_time += 60
                    elif event.key == pygame.K_LEFT:
                        elapsed_time -= 60
                    elif event.key == pygame.K_0:
                        self.save_reduced_data()
            
            # Clear the screen
            self.screen.fill(self.WHITE)
            
            # Calculate elapsed time
            elapsed_time += (pygame.time.get_ticks() - previous_frame_time) * simulation_speed
            previous_frame_time = pygame.time.get_ticks()
            
            # Determine current image index
            self.current_image_index = min(
                int(elapsed_time / (self.joystick_data[-1]['timestamp'].astype('int64') - self.joystick_data[0]['timestamp'].astype('int64')) * len(self.images)), 
                len(self.images) - 1
            )
            
            # Draw current image
            if self.images:
                self.screen.blit(self.images[self.current_image_index], (0, self.SCREEN_HEIGHT // 2))
            
            # Draw simulation area border
            pygame.draw.rect(self.screen, self.GRAY, 
                             (0, self.SCREEN_HEIGHT // 2, self.SCREEN_WIDTH, self.SCREEN_HEIGHT // 2), 2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 8, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 3, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 5.6, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2.7, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 1.3, self.SCREEN_HEIGHT // 8, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             (self.SCREEN_WIDTH // 1.3, self.SCREEN_HEIGHT // 3, self.SCREEN_WIDTH // 25, self.SCREEN_HEIGHT // 25), 
                             2)
            
            # Draw trajectory
            trajectory_points = []
            for point in self.trajectory:
                # Check if this point should be drawn based on elapsed time
                if point['timestamp'].astype('int64') - self.joystick_data[0]['timestamp'].astype('int64') <= elapsed_time:
                    screen_x, screen_y = self.convert_to_screen_coords(point['x'], point['y'])
                    trajectory_points.append((screen_x, screen_y + self.SCREEN_HEIGHT // 2))
            
            # Draw trajectory line
            if len(trajectory_points) > 1:
                pygame.draw.lines(self.screen, self.BLUE, False, trajectory_points, 2)

            # Draw current position
            if trajectory_points:
                current_pos = trajectory_points[-1]
                pygame.draw.circle(self.screen, self.RED, current_pos, 5)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            
            # End simulation if all points have been drawn
            if len(trajectory_points) >= len(self.trajectory):
                running = False
        
        pygame.quit()

def main():
    outputs_path = "D:/bachelor arbeit/outputs"
    for path in os.listdir(outputs_path):
        path = outputs_path + "/" + path
        joystick_data = np.load(path+"/data.npy")

        # Create and run simulator
        simulator = VehicleTrajectorySimulator(joystick_data, path+"/images")
        simulator.run_simulation()

if __name__ == "__main__":
    main()