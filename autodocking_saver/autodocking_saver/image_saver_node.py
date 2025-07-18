#import debugpy
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from custom_msgs.msg import ImagePair # type: ignore
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from sensor_msgs.msg import CompressedImage
import cv2

from cv_bridge import CvBridge
import os
import time
from PIL import Image

from .open_waters import predict_ros_adaptation

class AutodockingSaver(Node):
    def __init__(self):
        super().__init__('autodocking_saver')
        self.qos_profile = qos_profile_sensor_data

        # Declare the parameter 'save_image'
        self.declare_parameter('save_image', Parameter.Type.BOOL)
        self.save_image_enabled = self.get_parameter('save_image').value

        self.subscrier_left_ = self.create_subscription(
            CompressedImage,
            '/cam_left/frame',
            self.left_callback,
            self.qos_profile
        )
        self.bridge = CvBridge()
        self.processing = False  # Flag to indicate if processing is ongoing
        self.image_counter = 1
        self.output_dir = f'/home/ubuntu/autodocking-ros2/camera_images' # Where to save the processed image
        os.makedirs(self.output_dir, exist_ok=True)
    
    def left_callback(self, msg):
        
        if self.processing:
            # Skip this callback if still processing the previous image
            return
        
        self.processing = True
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * pow(10,-9)
        id = msg.header.frame_id

        #self.get_logger().info(f"Frame {id} at {timestamp}, now: {time.time()}")
        time_0 = time.time()
        try:
            np_arr_left = np.frombuffer(msg.data, dtype=np.uint8)
            frame_left = cv2.imdecode(np_arr_left, cv2.IMREAD_COLOR)
            if frame_left is None:
                    self.get_logger().error("Can't decode frame")
                    return
            
            cv_image = frame_left #self.bridge.imgmsg_to_cv2(frame_left, desired_encoding='bgr8') # Convert the image from a ROS2 message to a cv2 image
            image_to_process = self.cv_to_pil(cv_image) # Convert the cv2 image to a PIL image
            self.save_image(image_to_process) # Save the image in a folder
            self.process_image(image_to_process)
            self.processing = False
            
        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")
            self.processing = False

        #self.get_logger().info(f"Decoded frame in { (time.time() - time_0 )*1000 }ms")
        delta_receive = round((time.time() - timestamp)*1000, 2)
        delta_process = round((time.time() - time_0 )*1000 , 2)
        #self.get_logger().info(f"{id}, R: {delta_receive}ms, P: {delta_process}ms")
        

    def process_image(self, image):
        start = time.time()
        predict_ros_adaptation.predict(image, self.save_image_enabled,f"{self.output_dir}/out-{self.image_counter}.jpg") # Call the function that will process the image
        end = time.time()
        elapsed = end - start
        self.get_logger().info(f"Inference time (image {self.image_counter}): {elapsed:.2f}s\n") # Show the time it took to process the image
        self.image_counter += 1

    # Function to convert the image from cv2 to a PIL image
    def cv_to_pil(self, cv2_image):
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    
    # Function to save the image locally inside a folder
    def save_image(self, image):
        # Skip this function execution if the shouldn't be saved
        if not self.save_image_enabled:
            return
        
        # Proceed to save the image if it should be saved
        try:
            image_path = os.path.join(self.output_dir, f'{self.image_counter}.jpg')
            image.save(image_path)
            self.get_logger().info(f'Saved image {image_path}')
        except Exception as e:
            self.get_logger().error(f'Error saving image: {e}')

def main(args=None):
    rclpy.init(args=args)
    #debug()

    node = AutodockingSaver()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    
'''def debug():
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    debugpy.breakpoint()
'''
