import os
import logging
import time
import numpy as np
import cv2
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
COVID_PREDICTION_LABELS = ['Covid-19', 'Non Covid-19']
IMAGE_SIZE = (64, 64)
MODEL_CONFIGS = [
    {
        'name': 'VGG16',
        'path': 'predictor/model_weights/VGG16/VGG16_Model.h5'
    },
    {
        'name': 'ResNet50',
        'path': 'predictor/model_weights/ResNet50/ResNet50_Model.h5'
    },
    {
        'name': 'Xception',
        'path': 'predictor/model_weights/Xception/Xception_Model.h5'
    }
]

def prepare_image(image_path, target_size=IMAGE_SIZE):
    """
    Prepare image for model prediction with robust error handling.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size for resizing
    
    Returns:
        numpy.ndarray or None: Preprocessed image array
    """
    try:
        # Use Keras image loading to handle various image formats
        img = load_img(image_path, target_size=target_size)
        
        # Convert image to numpy array
        img_array = img_to_array(img)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image prepared successfully. Shape: {img_array.shape}")
        return img_array
    
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        return None

def load_and_predict_model(model_path, image_array):
    """
    Load a model and make a prediction with comprehensive error handling.
    
    Args:
        model_path (str): Path to the model file
        image_array (numpy.ndarray): Preprocessed image array
    
    Returns:
        tuple: (prediction index, confidence score, execution time)
    """
    if image_array is None:
        logger.warning("Cannot predict: Invalid image array")
        return None, None, None
    
    try:
        # Construct full absolute path
        full_model_path = os.path.join(settings.BASE_DIR, model_path)
        
        # Check if model file exists
        if not os.path.exists(full_model_path):
            logger.error(f"Model file not found: {full_model_path}")
            return None, None, None
        
        # Start timing
        start_time = time.time()
        
        # Load model
        model = load_model(full_model_path)
        
        # Predict
        predictions = model.predict(image_array)
        prediction_idx = np.argmax(predictions[0])
        confidence_score = float(np.amax(predictions[0]))
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"Model prediction successful. Prediction: {prediction_idx}, Confidence: {confidence_score}")
        
        return prediction_idx, confidence_score, execution_time
    
    except Exception as e:
        logger.error(f"Model prediction error for {model_path}: {e}")
        return None, None, None

def clear_media_directory():
    """
    Clear the media directory safely.
    """
    media_dir = os.path.join(settings.BASE_DIR, 'media')
    try:
        for filename in os.listdir(media_dir):
            file_path = os.path.join(media_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error accessing media directory: {e}")

def index(request):
    """
    Main view for COVID-19 image prediction.
    
    Handles image upload and runs predictions across multiple models.
    """
    response = {}
    
    if request.method == "POST":
        # Clear previous media files
        clear_media_directory()
        
        # Handle file upload
        try:
            uploaded_image = request.FILES.get('ImgFile')
            if not uploaded_image:
                logger.warning("No image file uploaded")
                return render(request, 'index.html', {'error': 'No image uploaded'})
            
            # Save uploaded image
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            image_path = fs.path(filename)
            
            # Prepare response data
            response['image'] = f"../media/{filename}"
            response.update({
                'table': 'table',
                'col0': ' ',
                'col1': 'VGG16',
                'col2': 'ResNet50',
                'col3': 'Xception',
                'row1': 'Results',
                'row2': 'Confidence Score',
                'row3': 'Prediction Time (s)'
            })
            
            # Prepare image for prediction
            prepared_image = prepare_image(image_path)
            
            # Run predictions for each model
            results = {}
            for model_config in MODEL_CONFIGS:
                model_name = model_config['name']
                model_path = model_config['path']
                
                # Predict
                pred_idx, confidence, exec_time = load_and_predict_model(model_path, prepared_image)
                
                # Process results
                if pred_idx is not None:
                    results[model_name] = {
                        'prediction': COVID_PREDICTION_LABELS[pred_idx],
                        'confidence': confidence,
                        'time': exec_time
                    }
                else:
                    results[model_name] = {
                        'prediction': 'Model Error',
                        'confidence': 'N/A',
                        'time': 'N/A'
                    }
            
            # Populate response with results
            response.update({
                'v_pred': results['VGG16']['prediction'],
                'r_pred': results['ResNet50']['prediction'],
                'x_pred': results['Xception']['prediction'],
                
                'v_cf': results['VGG16']['confidence'],
                'r_cf': results['ResNet50']['confidence'],
                'x_cf': results['Xception']['confidence'],
                
                'v_time': results['VGG16']['time'],
                'r_time': results['ResNet50']['time'],
                'x_time': results['Xception']['time']
            })
            
            logger.info("Prediction completed successfully")
            return render(request, 'index.html', response)
        
        except Exception as e:
            logger.error(f"Unexpected error in prediction process: {e}")
            return render(request, 'index.html', {'error': 'An unexpected error occurred'})
    
    # GET request handling
    return render(request, 'index.html')