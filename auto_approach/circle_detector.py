import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_circles(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red and black
    # Red has two ranges in HSV (wraps around)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Black range
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    # Create masks for red and black
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Apply morphological operations to clean up the masks
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    
    # Find circles using HoughCircles
    circles_data = []
    
    # Detect circles in red mask
    red_circles = cv2.HoughCircles(mask_red, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=5, maxRadius=100)
    
    if red_circles is not None:
        red_circles = np.uint16(np.around(red_circles))
        for circle in red_circles[0]:
            x, y, radius = circle
            circles_data.append({
                'color': 'red',
                'x': x,
                'y': y,
                'radius': radius
            })
    
    # Detect circles in black mask
    black_circles = cv2.HoughCircles(mask_black, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                     param1=50, param2=30, minRadius=5, maxRadius=100)
    
    if black_circles is not None:
        black_circles = np.uint16(np.around(black_circles))
        for circle in black_circles[0]:
            x, y, radius = circle
            circles_data.append({
                'color': 'black',
                'x': x,
                'y': y,
                'radius': radius
            })
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image with detected circles
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Draw detected circles
    img_with_circles = img_rgb.copy()
    for circle in circles_data:
        color = (255, 0, 0) if circle['color'] == 'red' else (0, 0, 0)
        cv2.circle(img_with_circles, (circle['x'], circle['y']), circle['radius'], color, 2)
        cv2.circle(img_with_circles, (circle['x'], circle['y']), 2, color, 3)
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_with_circles)
    plt.title('Detected Circles')
    plt.axis('off')
    
    # Show masks
    plt.subplot(1, 3, 3)
    combined_mask = cv2.bitwise_or(mask_red, mask_black)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Combined Mask (Red + Black)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('circle_detection_results.png')
    plt.show()
    
    return circles_data

def save_to_csv(circles_data, output_file='circle_centers.csv'):
    # Convert to DataFrame
    df = pd.DataFrame(circles_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(circles_data)} circle centers to {output_file}")
    
    return df

# Main execution
if __name__ == "__main__":
    image_path = "GSN-256.png"
    
    # Detect circles
    circles = detect_circles(image_path)
    
    # Save to CSV
    df = save_to_csv(circles)
    
    # Display the results
    print("\nDetected Circle Centers:")
    print(df) 