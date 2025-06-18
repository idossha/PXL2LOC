import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, feature
from skimage.transform import hough_circle, hough_circle_peaks

def preprocess_image(img):
    """Enhanced preprocessing for better circle detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered, enhanced, gray

def detect_circles_comprehensive(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Preprocess the image
    filtered, enhanced, gray = preprocess_image(img)
    
    all_circles = []
    
    # Method 1: Standard HoughCircles with various parameters
    print("Method 1: HoughCircles detection...")
    for minDist in [10, 15, 20]:
        for param2 in [10, 15, 20, 25]:
            circles = cv2.HoughCircles(
                filtered,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=minDist,
                param1=50,
                param2=param2,
                minRadius=3,
                maxRadius=50
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0]:
                    x, y, r = circle
                    if 10 < x < width-10 and 10 < y < height-10:  # Avoid edge artifacts
                        all_circles.append((x, y, r))
    
    # Method 2: Edge detection + contour finding
    print("Method 2: Edge + contour detection...")
    edges = cv2.Canny(filtered, 30, 100)
    
    # Dilate edges to connect broken circles
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_cleaned = cv2.erode(edges_dilated, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 3000:  # Reasonable area for electrodes
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # Reasonably circular
                    all_circles.append((int(x), int(y), int(radius)))
    
    # Method 3: Blob detection
    print("Method 3: Blob detection...")
    # Invert image for blob detection
    inverted = 255 - filtered
    
    # Setup SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 2000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = False
    params.filterByInertia = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        all_circles.append((x, y, radius))
    
    # Method 4: Template matching for circular patterns
    print("Method 4: Template matching...")
    # Create circular templates of different sizes
    for radius in range(5, 25, 2):
        template = np.zeros((radius*2+1, radius*2+1), dtype=np.uint8)
        cv2.circle(template, (radius, radius), radius, 255, -1)
        
        # Match template
        result = cv2.matchTemplate(filtered, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt[0] + radius, pt[1] + radius
            all_circles.append((x, y, radius))
    
    # Remove duplicates and cluster nearby detections
    print("Clustering and filtering detections...")
    unique_circles = cluster_circles(all_circles, min_distance=10)
    
    # Determine colors of detected circles
    circles_with_color = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for (x, y, r) in unique_circles:
        # Sample color from the center of the circle
        if 0 <= y < height and 0 <= x < width:
            center_hsv = hsv[y, x]
            center_bgr = img[y, x]
            
            # Determine if red, black, or other
            color = classify_electrode_color(center_hsv, center_bgr)
            
            circles_with_color.append({
                'x': x,
                'y': y,
                'radius': r,
                'color': color
            })
    
    return circles_with_color

def cluster_circles(circles, min_distance=10):
    """Remove duplicate circles that are too close together"""
    if not circles:
        return []
    
    circles = list(set(circles))  # Remove exact duplicates
    circles.sort(key=lambda c: (c[1], c[0]))  # Sort by y, then x
    
    filtered = []
    for circle in circles:
        x, y, r = circle
        
        # Check if this circle is too close to any already accepted circle
        too_close = False
        for fx, fy, fr in filtered:
            distance = np.sqrt((x - fx)**2 + (y - fy)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered.append(circle)
    
    return filtered

def classify_electrode_color(hsv_pixel, bgr_pixel):
    """Classify electrode color based on HSV and BGR values"""
    h, s, v = hsv_pixel
    b, g, r = bgr_pixel
    
    # Check for red (considering both ranges in HSV)
    if ((h <= 10 or h >= 170) and s > 50 and v > 50) or (r > 150 and g < 100 and b < 100):
        return 'red'
    # Check for black/dark gray
    elif v < 80 or (r < 80 and g < 80 and b < 80):
        return 'black'
    # Check for pink/light red
    elif (h <= 20 or h >= 160) and s > 20 and v > 100:
        return 'red'
    else:
        # Default to classification based on brightness
        return 'black' if v < 127 else 'red'

def visualize_results(img_path, circles_data):
    """Create comprehensive visualization of results"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Detected electrodes with colors
    img_annotated = img_rgb.copy()
    for i, circle in enumerate(circles_data):
        # Draw circle
        color = (255, 0, 0) if circle['color'] == 'red' else (0, 0, 0)
        cv2.circle(img_annotated, (circle['x'], circle['y']), circle['radius'], color, 2)
        cv2.circle(img_annotated, (circle['x'], circle['y']), 2, color, -1)
        
        # Add number for first 30
        if i < 30:
            cv2.putText(img_annotated, str(i+1), 
                       (circle['x']+5, circle['y']-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    plt.subplot(2, 3, 2)
    plt.imshow(img_annotated)
    plt.title(f'Detected {len(circles_data)} Electrodes')
    plt.axis('off')
    
    # Preprocessing visualization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered, enhanced, _ = preprocess_image(img)
    
    plt.subplot(2, 3, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    
    # Edge detection
    edges = cv2.Canny(filtered, 30, 100)
    plt.subplot(2, 3, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    # Statistics
    plt.subplot(2, 3, 5)
    colors = [c['color'] for c in circles_data]
    color_counts = pd.Series(colors).value_counts()
    plt.bar(color_counts.index, color_counts.values)
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.title('Color Distribution')
    for i, (color, count) in enumerate(color_counts.items()):
        plt.text(i, count + 1, str(count), ha='center')
    
    # Spatial distribution
    plt.subplot(2, 3, 6)
    x_coords = [c['x'] for c in circles_data]
    y_coords = [c['y'] for c in circles_data]
    colors_plot = ['red' if c['color'] == 'red' else 'black' for c in circles_data]
    plt.scatter(x_coords, y_coords, c=colors_plot, s=50, alpha=0.7, edgecolors='gray')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Spatial Distribution')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_annotated

def save_to_csv(circles_data, filename='electrode_centers_comprehensive.csv'):
    """Save results to CSV with additional metadata"""
    df = pd.DataFrame(circles_data)
    
    # Add electrode numbering
    df['electrode_number'] = range(1, len(df) + 1)
    
    # Sort by y-coordinate, then x-coordinate for consistent ordering
    df = df.sort_values(['y', 'x']).reset_index(drop=True)
    df['electrode_number'] = range(1, len(df) + 1)
    
    # Calculate additional features
    df['area'] = np.pi * df['radius']**2
    
    # Reorder columns
    df = df[['electrode_number', 'color', 'x', 'y', 'radius', 'area']]
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} electrode centers to {filename}")
    
    return df

# Main execution
if __name__ == "__main__":
    image_path = "GSN-256.png"
    
    print("Running comprehensive electrode detection...")
    print("This may take a moment as multiple detection methods are being used...\n")
    
    # Detect electrodes
    circles = detect_circles_comprehensive(image_path)
    
    print(f"\nTotal electrodes detected: {len(circles)}")
    
    # Save results
    df = save_to_csv(circles)
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_results(image_path, circles)
    
    # Print summary statistics
    print("\nDetection Summary:")
    print(f"Total electrodes: {len(circles)}")
    print(f"Red electrodes: {sum(1 for c in circles if c['color'] == 'red')}")
    print(f"Black electrodes: {sum(1 for c in circles if c['color'] == 'black')}")
    
    if len(circles) < 256:
        print(f"\nWarning: Expected 256 electrodes but found {len(circles)}")
        print("Consider adjusting detection parameters or checking image quality")
    
    print("\nFirst 10 detected electrodes:")
    print(df.head(10))
    print("\nLast 10 detected electrodes:")
    print(df.tail(10)) 