import numpy as np
import pygame
import math
import os
from tqdm import tqdm
import csv
import scipy.interpolate
import time

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
        self.AREA_HEIGHT = 1.0  # 1 meter tall
        
        # Screen dimensions
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        
        # Image display area
        self.IMAGE_HEIGHT = screen_height // 2
        self.IMAGE_WIDTH = screen_width

        # Load images
        self.images = self.load_images(image_directory)
        
        # Vehicle and simulation parameters
        self.tau = 0.05
        self.rolling_resistance = 0.15
        self.static_threshold = 0.23
        self.speed_coefficient = 1
        
        # Initial vehicle position (30cm from one long side)
        self.initial_x = 0.13
        self.initial_y = 0.335
        
        # Simulation state
        self.current_x = self.initial_x
        self.current_y = self.initial_y
        self.current_theta = 0
        
        # Trajectory data
        self.joystick_data = joystick_data
        self.trajectory = self.compute_trajectory()
        
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
    
    def compute_trajectory(self):
        # Given data: coefficients and corresponding velocities
        coefficients = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        avg_velocity = np.array([0.0, 0.07, 0.13, 0.19, 0.26, 0.3, 0.38, 0.43, 0.48, 0.55])

        # Create an interpolation function
        f = scipy.interpolate.interp1d(coefficients, avg_velocity, kind='linear', fill_value='extrapolate')
    
        def f_velocity(k, V_prev):
            if abs(k) < self.static_threshold and V_prev == 0:
                return 0  # Not enough power to start

            V = self.speed_coefficient * np.sign(k) * f(abs(k))  # Normal speed calculation

            return V

        # Initial state
        x, y, theta = self.initial_x, self.initial_y, 0
        wheelbase = 0.1
        data = self.joystick_data
        trajectory = [{
            'x': x,
            'y': y,
            'timestamp': np.float64(0)
        }]  # Initial position with first timestamp

        V_l_prev, V_r_prev = 0, 0

        for i in range(1, len(data)):
            point = data[i]
            k_l, k_r = self.calculate_motor_speeds(point['x'], point['y'])
            t = point['timestamp'].astype('float64')
            prev_t = data[i - 1]['timestamp'].astype('float64')

            # Compute velocity
            V_l_target = f_velocity(k_l, V_l_prev)
            V_r_target = f_velocity(k_r, V_r_prev)

            # Compute time step
            dt = (t - prev_t) / 1000.0  # Convert ms to seconds

            # Simulate acceleration response
            V_l = V_l_prev + (V_l_target - V_l_prev) / self.tau * dt
            V_r = V_r_prev + (V_r_target - V_r_prev) / self.tau * dt

            # Simulate rolling resistance
            V_l = V_l - np.sign(V_l) * self.rolling_resistance * dt
            V_r = V_r - np.sign(V_r) * self.rolling_resistance * dt

            # Compute linear
            V = (V_l + V_r) / 2
            
            # Compute angular velocity
            omega = (V_r - V_l) / (2 * wheelbase)

            # Update state
            V_l_prev, V_r_prev = V_l, V_r

            # Update position
            theta = theta + omega * dt
            x += V * dt * np.cos(theta)
            y += V * dt * np.sin(theta)

            trajectory.append({
                'x': x,
                'y': y,
                'timestamp': t
            })

        return trajectory
    
    def hyperparameter(self):
        print("starting hyperparameter search")
        matching_parameters = []
        for tau in tqdm([wheel_base / 100 for wheel_base in range(1, 30, 1)]):
            for rolling_resistance in tqdm([motor_efficiency / 100 for motor_efficiency in range(0, 30, 1)], leave=False):
                for speed_coefficient in tqdm([speed_threashold / 100 for speed_threashold in range(50, 150, 1)], leave=False):
                    self.tau = tau
                    self.rolling_resistance = rolling_resistance
                    self.speed_coefficient = speed_coefficient
                    trajectory = self.compute_trajectory()

                    failed = False

                    total_time = self.joystick_data[-1]['timestamp'].astype('int64') - self.joystick_data[0]['timestamp'].astype('int64')

                    for index in range(int(1200 / total_time * len(self.images)), int(1300 / total_time * len(self.images))): # checking if vehicle goes in between first pair of obstacles
                        y = trajectory[index]['y']
                        if y < 0.36 or y > 0.6:
                            failed = True
                            break
                    
                    if failed:
                        continue

                    for index in range(int(2700 / total_time * len(self.images)), int(2900 / total_time * len(self.images))): # checking if vehicle goes in between second pair of obstacles
                        y = trajectory[index]['y']
                        if y < 0.44 or y > 0.7:
                            failed = True
                            break
                    
                    if failed:
                        continue

                    for index in range(int(4000 / total_time * len(self.images)), int(4200 / total_time * len(self.images))): # checking if vehicle goes in between third pair of obstacles
                        y = trajectory[index]['y']
                        if y < 0.36 or y > 0.6:
                            failed = True
                            break

                    if failed:
                        continue

                    for index in range(int(4600 / total_time * len(self.images)), int(5400 / total_time * len(self.images))): # checking if vehicle stays inside of the arena while turning
                        x = trajectory[index]['x']
                        if x < 1.5 or x > 2:
                            failed = True
                            break

                    if not failed:
                        params = {
                            'tau': tau,
                            'rolling_resistance': rolling_resistance,
                            'speed_coefficient': speed_coefficient
                        }
                        print("found passing params:", params)
                        matching_parameters.append(params)

        print("best parameters:")
        print(matching_parameters)

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
                             ((0.5 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.36 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             ((0.5 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.66 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             ((0.98 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.26 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             ((0.98 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.56 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             ((1.46 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.36 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
                             2)
            pygame.draw.rect(self.screen, 
                             self.RED, 
                             ((1.46 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.66 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2, (0.04 / self.AREA_WIDTH) * self.SCREEN_WIDTH, (0.04 / self.AREA_HEIGHT) * self.SCREEN_HEIGHT / 2), 
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
    outputs_path = "/Users/andrewyarotskyi/Downloads/outputs"
    for path in os.listdir(outputs_path):
        path = outputs_path + "/" + path
        joystick_data = np.load(path+"/data.npy")

        # Create and run simulator
        simulator = VehicleTrajectorySimulator(joystick_data, path+"/images")
        simulator.hyperparameter()

if __name__ == "__main__":
    main()