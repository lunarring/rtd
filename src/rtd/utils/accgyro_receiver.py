import socket
import threading
import time
import math

class AccelerometerGyroReceiver:
    def __init__(self, host='', port=7777):
        self.host = host
        self.port = port
        self.last_accelerometer = None  # (timestamp, ax, ay, az)
        self.last_gyroscope = None      # (timestamp, gx, gy, gz)
        self.integrated_angle_z = 0.0   # Accumulated angle (in degrees) around Z-axis
        self.integrated_angle_x = 0.0   # Accumulated angle (in degrees) around X-axis
        self.integrated_angle_y = 0.0   # Accumulated angle (in degrees) around Y-axis
        self.last_gyro_timestamp = None
        self.gyro_offset_z = None       # Offset set to the first integrated gyro value per session
        self.gyro_offset_x = None       # Offset for X-axis rotation
        self.gyro_offset_y = None       # Offset for Y-axis rotation
        self.running = True
        
        # For angular velocity calculation
        self.last_angle_z = 0.0
        self.last_angle_z_timestamp = None
        self.angular_velocity_z = 0.0

        # Window for the last 3 accelerometer samples (each sample is a tuple: (ax, ay, az))
        self.acc_window = []
        
        # For velocity calculation
        self.last_acc_timestamp = None
        self.velocity = [0.0, 0.0, 0.0]  # vx, vy, vz
        self.velocity_damping = 0.95     # Damping factor to prevent drift (0.95 means 5% reduction per update)
        
        # For position calculation
        self.last_vel_timestamp = None
        self.position = [0.0, 0.0, 0.0]  # px, py, pz
        self.position_damping = 0.98     # Damping factor for position (0.98 means 2% reduction per update)

        # Start the background thread for receiving sensor data
        self.thread = threading.Thread(target=self._server_thread, daemon=True)
        self.thread.start()

    def _server_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            print(f"Server listening on port {self.port}...")
            while self.running:
                try:
                    print("Waiting for connection...")
                    conn, addr = s.accept()
                    print(f"Connected by {addr}")
                    # Reset gyroscope integration and offset for new session
                    self.integrated_angle_z = 0.0
                    self.integrated_angle_x = 0.0
                    self.integrated_angle_y = 0.0
                    self.last_gyro_timestamp = None
                    self.gyro_offset_z = None
                    self.gyro_offset_x = None
                    self.gyro_offset_y = None
                    # Reset velocity calculation
                    self.last_acc_timestamp = None
                    self.velocity = [0.0, 0.0, 0.0]
                    # Reset position calculation
                    self.last_vel_timestamp = None
                    self.position = [0.0, 0.0, 0.0]
                    # Reset angular velocity calculation
                    self.last_angle_z = 0.0
                    self.last_angle_z_timestamp = None
                    self.angular_velocity_z = 0.0
                    with conn:
                        buffer = ""
                        while self.running:
                            try:
                                data = conn.recv(1024)
                                if not data:
                                    print("Client disconnected.")
                                    break
                                buffer += data.decode()
                                # Process complete lines terminated by newline
                                while "\n" in buffer:
                                    line, buffer = buffer.split("\n", 1)
                                    self._process_line(line.strip())
                            except Exception as e:
                                print("Error during data reception:", e)
                                break
                except Exception as e:
                    print("Server error:", e)
                    break

    def _process_line(self, line):
        """
        Processes a line of sensor data.
        Expected format: SENSOR_TYPE,timestamp,value1,value2,value3
        For example:
          "GYRO,1234567890,0.01,0.02,0.03"
          "ACC,1234567890,9.81,0.01,-0.05"
        """
        parts = line.split(',')
        if len(parts) != 5:
            return
        sensor_type = parts[0]
        try:
            timestamp = float(parts[1])
            v1 = float(parts[2])
            v2 = float(parts[3])
            v3 = float(parts[4])
        except ValueError:
            return

        if sensor_type == "ACC":
            self.last_accelerometer = (timestamp, v1, v2, v3)
            # Add new accelerometer sample to the window
            self.acc_window.append((v1, v2, v3))
            # Ensure the window size remains 3 samples
            if len(self.acc_window) > 3:
                self.acc_window.pop(0)
                
            # Calculate mean acceleration over the window
            if len(self.acc_window) > 0:
                mean_acc = self._calculate_mean_acceleration()
                
                # Update velocity based on mean acceleration
                if self.last_acc_timestamp is not None:
                    dt = (timestamp - self.last_acc_timestamp) / 1e9  # dt in seconds (Android timestamps are in ns)
                    
                    # Apply damping to current velocity (reduces drift)
                    self.velocity = [v * self.velocity_damping for v in self.velocity]
                    
                    # Integrate mean acceleration to get velocity
                    self.velocity[0] += mean_acc[0] * dt
                    self.velocity[1] += mean_acc[1] * dt
                    self.velocity[2] += mean_acc[2] * dt
                    
                    # Update position based on velocity
                    if self.last_vel_timestamp is not None:
                        vel_dt = (timestamp - self.last_vel_timestamp) / 1e9  # dt in seconds
                        
                        # Apply damping to current position (pushes towards zero)
                        self.position = [p * self.position_damping for p in self.position]
                        
                        # Integrate velocity to get position
                        self.position[0] += self.velocity[0] * vel_dt
                        self.position[1] += self.velocity[1] * vel_dt
                        self.position[2] += self.velocity[2] * vel_dt
                
                self.last_vel_timestamp = timestamp
            self.last_acc_timestamp = timestamp
            
        elif sensor_type == "GYRO":
            self.last_gyroscope = (timestamp, v1, v2, v3)
            # Integrate the rotation about all three axes
            if self.last_gyro_timestamp is not None:
                dt = (timestamp - self.last_gyro_timestamp) / 1e9  # dt in seconds (Android timestamps are in ns)
                # Convert angular velocity (rad/s) to degrees
                delta_angle_x = v1 * dt * (180.0 / math.pi)
                delta_angle_y = v2 * dt * (180.0 / math.pi)
                delta_angle_z = v3 * dt * (180.0 / math.pi)
                self.integrated_angle_x += delta_angle_x
                self.integrated_angle_y += delta_angle_y
                self.integrated_angle_z += delta_angle_z
            self.last_gyro_timestamp = timestamp

            # On the first gyro event after connection, store the offsets
            if self.gyro_offset_z is None:
                self.gyro_offset_z = self.integrated_angle_z
                self.gyro_offset_x = self.integrated_angle_x
                self.gyro_offset_y = self.integrated_angle_y
            
            # Calculate angular velocity for Z axis
            current_angle_z = self.integrated_angle_z - (self.gyro_offset_z or 0)
            if self.last_angle_z_timestamp is not None:
                dt = (timestamp - self.last_angle_z_timestamp) / 1e9  # dt in seconds
                if dt > 0:  # Avoid division by zero
                    self.angular_velocity_z = (current_angle_z - self.last_angle_z) / dt
            
            self.last_angle_z = current_angle_z
            self.last_angle_z_timestamp = timestamp
    
    def _calculate_mean_acceleration(self):
        """
        Calculate the mean acceleration values from the window of samples.
        """
        if not self.acc_window:
            return (0.0, 0.0, 0.0)
            
        sum_x = sum(sample[0] for sample in self.acc_window)
        sum_y = sum(sample[1] for sample in self.acc_window)
        sum_z = sum(sample[2] for sample in self.acc_window)
        n = len(self.acc_window)
        
        return (sum_x / n, sum_y / n, sum_z / n)

    def get_last_gyroscope_values(self):
        """
        Returns the calibrated integrated rotation angles (in degrees) around all three axes:
        X, Y, and Z, where the calibration subtracts the first received value for each axis.
        Returns a tuple (angle_x, angle_y, angle_z).
        
        Axes definition:
          - Z: axis orthogonal to the screen (pointing out)
          - Y: axis along the long side of the smartphone
          - X: axis along the short side of the smartphone
        """
        angle_x = self.integrated_angle_x
        angle_y = self.integrated_angle_y
        angle_z = self.integrated_angle_z
        
        if self.gyro_offset_x is not None:
            angle_x -= self.gyro_offset_x
        if self.gyro_offset_y is not None:
            angle_y -= self.gyro_offset_y
        if self.gyro_offset_z is not None:
            angle_z -= self.gyro_offset_z
            
        return (angle_x, angle_y, angle_z)

    def get_last_angular_velocity_z(self):
        """
        Returns the angular velocity around the Z-axis in degrees per second.
        This is calculated as the numerical derivative of the calibrated gyroscope angle.
        
        Returns:
            float: Angular velocity in degrees per second, or 0.0 if no data is available yet.
        """
        return self.angular_velocity_z

    def get_last_accelerometer_values(self):
        """
        Returns the most recent accelerometer values as a tuple (ax, ay, az) after subtracting the mean of
        the last 3 samples from the latest sample.
        Axes definition:
          - Z: axis orthogonal to the screen,
          - Y: axis along the long side of the smartphone,
          - X: axis along the short side.
        """
        if self.last_accelerometer is not None and self.acc_window:
            # Calculate the mean over the accelerometer window
            mean_acc = self._calculate_mean_acceleration()
            
            _, ax, ay, az = self.last_accelerometer
            # Return the mean acceleration instead of the raw values
            return mean_acc
        return None
        
    def get_last_velocities(self):
        """
        Returns the current velocity components as a tuple (vx, vy, vz) calculated by
        integrating acceleration values over time with damping to prevent drift.
        Returns None if no acceleration data has been received yet.
        """
        if self.last_accelerometer is not None:
            return tuple(self.velocity)
        return None
    
    def get_last_positions(self):
        """
        Returns the current position components as a tuple (px, py, pz) calculated by
        integrating velocity values over time with damping to push positions toward zero.
        Returns None if no acceleration data has been received yet.
        """
        if self.last_accelerometer is not None:
            return tuple(self.position)
        return None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    from collections import deque
    
    sr = AccelerometerGyroReceiver(host='', port=7777)
    
    # Create a deque to store the last 100 velocity samples
    vx_history = deque(maxlen=100)
    
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, plot_ax = plt.subplots(figsize=(10, 4))
    line, = plot_ax.plot([], [])
    plot_ax.set_ylim(-10, 10)  # Set reasonable y-axis limits for position (larger than velocity)
    plot_ax.set_title('X Component of Position')
    plot_ax.set_xlabel('Samples')
    plot_ax.set_ylabel('Position (X)')
    
    try:
        while True:
            angle_x, angle_y, angle_z = sr.get_last_gyroscope_values()
            acc_values = sr.get_last_accelerometer_values()
            vel_values = sr.get_last_velocities()
            pos_values = sr.get_last_positions()
            angular_vel_z = sr.get_last_angular_velocity_z()
            
            if acc_values is not None:
                acc_x, acc_y, acc_z = acc_values
                vx, vy, vz = vel_values if vel_values else (0, 0, 0)
                px, py, pz = pos_values if pos_values else (0, 0, 0)
                print(f"Rotation angles - X: {angle_x:.2f}, Y: {angle_y:.2f}, Z: {angle_z:.2f} deg, "
                      f"Angular Velocity Z: {angular_vel_z:.2f} deg/s, "
                      f"Acceleration X: {acc_x:.2f}, Y: {acc_y:.2f}, Z: {acc_z:.2f}, "
                      f"Velocity X: {vx:.2f}, Y: {vy:.2f}, Z: {vz:.2f}, "
                      f"Position X: {px:.2f}, Y: {py:.2f}, Z: {pz:.2f}")
                
                # Add the current position x to the history instead of velocity
                vx_history.append(px*50)
                
                # Update the plot
                line.set_xdata(range(len(vx_history)))
                line.set_ydata(list(vx_history))
                plot_ax.set_xlim(0, max(100, len(vx_history)))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                print("No accelerometer data yet...")
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Exiting...")
        sr.running = False
        sr.thread.join()
        plt.close()
