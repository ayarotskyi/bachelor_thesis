import cv2
import numpy as np

def edge_detection_pipeline(image_path):
    """
    A comprehensive edge detection pipeline for blurred images.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    
    Returns:
    --------
    dict
        Dictionary containing various edge detection results
    """
    # Read the image
    image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (400, 200))[100:, :]
    
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Results dictionary to store different edge detection methods
    results = {}
    
    # 1. Heavy Gaussian Blurring (for extremely blurry images)
    blurred = cv2.GaussianBlur(image, (15, 15), 10)
    
    # 2. Adaptive Histogram Equalization to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = blurred
    
    # 3. Canny Edge Detection with adaptive thresholding
    median_intensity = np.median(enhanced)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    
    canny_edges = cv2.Canny(enhanced, 
                             threshold1=lower_threshold, 
                             threshold2=upper_threshold, 
                             apertureSize=5)
    
    # 4. Sobel Edge Detection (X and Y directions)
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 5. Laplacian of Gaussian (LoG) Edge Detection
    log_edges = cv2.Laplacian(enhanced, cv2.CV_64F)
    log_normalized = cv2.normalize(np.abs(log_edges), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 6. Morphological Gradient for Edge Enhancement
    kernel = np.ones((5,5), np.uint8)
    morphological_gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    
    median_intensity = np.median(morphological_gradient)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    
    sobel_canny = cv2.Canny(morphological_gradient, 
                             threshold1=lower_threshold, 
                             threshold2=upper_threshold, 
                             apertureSize=5)
    
    
    # Store results
    results = {
        'original': image,
        'blurred': blurred,
        'enhanced': enhanced,
        'canny_edges': canny_edges,
        'sobel_edges': sobel_normalized,
        'log_edges': log_normalized,
        'morphological_edges': morphological_gradient,
        'sobel_canny': sobel_canny,
        'original_uncut': cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (400, 200))
    }
    
    return results

def visualize_results(results):
    """
    Visualize the results of edge detection pipeline.
    
    Parameters:
    -----------
    results : dict
        Dictionary of processed images
    """
    for name, img in results.items():
        # Resize large images to fit screen
        if img.shape[0] > 800 or img.shape[1] > 1200:
            scale = min(800/img.shape[0], 1200/img.shape[1])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite("samples/"+name+".png", img)
        # cv2.imshow(name, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    try:
        # Replace with the path to your blurry image
        image_path = "D:/bachelor arbeit/reduced_data/images/419.png"
        
        # Run edge detection pipeline
        edge_results = edge_detection_pipeline(image_path)
        
        # Visualize results
        visualize_results(edge_results)
    
    except Exception as e:
        print(f"An error occurred: {e}")