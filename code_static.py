import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calculate_area(w, h):
    return w * h

def determine_quadrant(center_x, center_y, reference_x, reference_y):
    if center_x <= reference_x:
        if center_y <= reference_y:
            return '2nd(-,+)'
        else:
            return '3rd(-,-)'
    else:
        if center_y <= reference_y:
            return '1st(+,+)'
        else:
            return '4th(+,-)'

def detect_yellow_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            marked_objects.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'centroid': (centroid_x, centroid_y),
                'area': area
            })

    return marked_objects

# Load the image
image_path = 'C:\\Users\\rohan\\Desktop\\python code\\python code_SADA\\color-detection-opencv-master\\images\\VG_15.jpg'
original_image = cv2.imread(image_path)

# Detect yellow objects and mark them with red bounding boxes
marked_objects = detect_yellow_color(original_image)
marked_image = original_image.copy()

# Sample data (angle in degrees and corresponding pressure in bar)
angles = np.array([224,171,116,61.84,8.45,-41])
pressures = np.array([0, 5, 10, 15, 20, 25 ])

# Reshape the data (required for scikit-learn)
angles = angles.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Fit the model to your data
model.fit(angles, pressures)

# Find the largest and smallest area objects
if marked_objects:
    largest_object = max(marked_objects, key=lambda x: x['area'])
    smallest_object = min(marked_objects, key=lambda x: x['area'])

    # Calculate the angle between the centroids of the largest and smallest objects (considered as absolute value)
    largest_centroid_x, largest_centroid_y = largest_object['centroid']
    smallest_centroid_x, smallest_centroid_y = smallest_object['centroid']

    angle_rad = math.atan2(smallest_centroid_y - largest_centroid_y, smallest_centroid_x - largest_centroid_x)
    angle_deg = abs(math.degrees(angle_rad))  # Absolute value of angle

    # Determine the quadrant of the smallest object relative to the largest object
    quadrant = determine_quadrant(smallest_centroid_x, smallest_centroid_y, largest_centroid_x, largest_centroid_y)

    # Adjust angles for the 3rd and 4th quadrants
    if quadrant == '3rd(-,-)':
        angle_deg = 360 - angle_deg
    elif quadrant == '4th(+,-)':
        angle_deg = -angle_deg

    # Reshape the angle for prediction
    known_angle = np.array([[angle_deg]])  # Reshape to match the input format

    # Predict pressure based on the linear regression model
    predicted_pressure = model.predict(known_angle)

    # Display and print the results
    print(f'Largest Object - Centroid: ({largest_centroid_x}, {largest_centroid_y}), Area: {largest_object["area"]:.2f}')
    print(f'Smallest Object - Centroid: ({smallest_centroid_x}, {smallest_centroid_y}), Area: {smallest_object["area"]:.2f}')
    print(f'Adjusted Angle between Largest and Smallest Objects: {angle_deg:.2f} degrees')
    print(f'Smallest Object Quadrant relative to Largest Object: {quadrant}')
    print(f'Predicted Pressure: {predicted_pressure[0]:.2f} bar')

    # Plot the regression line for negative slope with axes swapped
    plt.scatter(angles, pressures, color='blue', label='Sample Data')
    plt.plot(angles, model.predict(angles), color='red', linewidth=3, label='Regression Line (Negative Slope)')
    plt.title('Negative Slope Linear Regression')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Pressure (bar)')
    plt.legend()
    plt.show()

    # Draw bounding boxes around yellow objects and print x, y coordinates
    for obj in marked_objects:
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        centroid_x, centroid_y = obj['centroid']
        area = obj['area']

        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red bounding box
        cv2.circle(marked_image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Mark centroid
        cv2.putText(marked_image, f'Area: {area:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(marked_image, f'X: {centroid_x}, Y: {centroid_y}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(marked_image, f'Adjusted Angle: {angle_deg:.2f} degrees', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(marked_image, f'Smallest Object Quadrant: {quadrant}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(marked_image, f'Predicted Pressure: {predicted_pressure[0]:.2f} bar', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Marked Image', marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No marked objects found.")
