# AWS Lambda Document Region Classification Service with Continual Learning
# Serverless AI service for classifying document regions using DynamoDB

import os
import json
import logging
import hashlib
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

import numpy as np
import cv2
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DocumentRegionClassifier:
    """
    AI-powered document region classifier for AWS Lambda
    Classifies regions as: handwritten_signature, printed_text, or stamp
    Uses DynamoDB for data persistence and S3 for model storage
    """
    
    def __init__(self):
        # AWS configuration from environment variables
        self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        self.table_prefix = os.environ.get('TABLE_PREFIX', 'classifier')
        self.s3_bucket = os.environ.get('MODEL_S3_BUCKET', 'document-classifier-models')
        self.model_key = os.environ.get('MODEL_S3_KEY', 'models/classifier.joblib')
        self.scaler_key = os.environ.get('SCALER_S3_KEY', 'models/scaler.joblib')
        
        # Table names
        self.predictions_table = f"{self.table_prefix}_predictions"
        self.feedback_table = f"{self.table_prefix}_feedback"
        self.training_data_table = f"{self.table_prefix}_training_data"
        
        # Classification labels
        self.classes = ["handwritten_signature", "printed_text", "stamp"]
        
        # Model components
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # AWS clients (initialized lazily)
        self._dynamodb = None
        self._s3 = None
        self._predictions_table = None
        self._feedback_table = None
        self._training_data_table = None
        
        # Load model from S3
        self._load_model_from_s3()
        
        # If no model exists, create initial model
        if not self.is_trained:
            self._create_initial_model()
    
    @property
    def dynamodb(self):
        """Lazy initialization of DynamoDB resource"""
        if self._dynamodb is None:
            self._dynamodb = boto3.resource('dynamodb', region_name=self.aws_region)
        return self._dynamodb
    
    @property
    def s3(self):
        """Lazy initialization of S3 client"""
        if self._s3 is None:
            self._s3 = boto3.client('s3', region_name=self.aws_region)
        return self._s3
    
    @property
    def predictions_table(self):
        """Lazy initialization of predictions table"""
        if self._predictions_table is None:
            self._predictions_table = self.dynamodb.Table(self.predictions_table)
        return self._predictions_table
    
    @property
    def feedback_table(self):
        """Lazy initialization of feedback table"""
        if self._feedback_table is None:
            self._feedback_table = self.dynamodb.Table(self.feedback_table)
        return self._feedback_table
    
    @property
    def training_data_table(self):
        """Lazy initialization of training data table"""
        if self._training_data_table is None:
            self._training_data_table = self.dynamodb.Table(self.training_data_table)
        return self._training_data_table
    
    def _load_model_from_s3(self):
        """Load existing model from S3 if available"""
        try:
            # Download classifier
            classifier_obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self.model_key)
            classifier_data = classifier_obj['Body'].read()
            self.classifier = joblib.loads(classifier_data)
            
            # Download scaler
            scaler_obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self.scaler_key)
            scaler_data = scaler_obj['Body'].read()
            self.scaler = joblib.loads(scaler_data)
            
            self.is_trained = True
            logger.info("Model loaded successfully from S3")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info("No existing model found in S3")
            else:
                logger.error(f"Error loading model from S3: {e}")
            self.is_trained = False
    
    def _save_model_to_s3(self):
        """Save the trained model to S3"""
        try:
            # Save classifier
            classifier_data = joblib.dumps(self.classifier)
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=self.model_key,
                Body=classifier_data,
                ContentType='application/octet-stream'
            )
            
            # Save scaler
            scaler_data = joblib.dumps(self.scaler)
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=self.scaler_key,
                Body=scaler_data,
                ContentType='application/octet-stream'
            )
            
            logger.info("Model saved successfully to S3")
            
        except Exception as e:
            logger.error(f"Error saving model to S3: {e}")
            raise
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive features from document region image"""
        features = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Basic image properties
        height, width = gray.shape
        features.extend([height, width, height/width if width > 0 else 1])
        
        # Intensity statistics
        features.extend([
            np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
            np.median(gray), np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        
        # Edge detection features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width) if (height * width) > 0 else 0
        features.append(edge_density)
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.extend([
            len(contours),
            np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,
            np.mean([cv2.arcLength(c, True) for c in contours]) if contours else 0
        ])
        
        # Simplified texture features
        lbp_hist = np.zeros(10)  # Simplified LBP histogram
        if gray.size > 0:
            # Simple texture measure using local variance
            kernel = np.ones((3,3), np.float32) / 9
            filtered = cv2.filter2D(gray, -1, kernel)
            texture_var = np.var(gray - filtered)
            lbp_hist[0] = texture_var / 255.0  # Normalize
        
        features.extend(lbp_hist.tolist())
        
        # Gradient features
        if gray.size > 0:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude) if gradient_magnitude.size > 0 else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        # Morphological features
        if gray.size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            features.extend([
                np.mean(opened), np.mean(closed),
                np.sum(opened != gray) / (height * width) if (height * width) > 0 else 0,
                np.sum(closed != gray) / (height * width) if (height * width) > 0 else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Shape features
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if gray.size > 0 else gray
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            features.extend([
                w/h if h > 0 else 1,
                cv2.contourArea(largest_contour) / (w * h) if w * h > 0 else 0,
                cv2.arcLength(largest_contour, True) / (2 * (w + h)) if (w + h) > 0 else 0
            ])
        else:
            features.extend([1, 0, 0])
        
        # Ensure we have exactly the expected number of features
        while len(features) < 35:
            features.append(0.0)
        
        return np.array(features[:35], dtype=np.float32)
    
    def _create_initial_model(self):
        """Create an initial model with synthetic training data"""
        logger.info("Creating initial model with synthetic data...")
        
        # Generate synthetic training data
        synthetic_data = []
        synthetic_labels = []
        
        np.random.seed(42)  # For reproducibility
        
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(50):
                if class_name == "handwritten_signature":
                    # Signatures: organic, variable features
                    base_features = [100, 300, 3.0, 128, 45, 20, 200, 140, 120, 180, 0.15, 8, 250, 180, 0.1] + [0.05] * 10 + [35, 25, 80, 120, 125, 0.3, 0.2, 2.5, 0.6, 0.8]
                elif class_name == "printed_text":
                    # Printed text: uniform, structured features
                    base_features = [80, 400, 5.0, 180, 30, 50, 220, 170, 160, 200, 0.25, 15, 180, 120, 0.15] + [0.08] * 10 + [45, 30, 100, 160, 165, 0.2, 0.15, 3.2, 0.8, 0.9]
                else:  # stamp
                    # Stamps: bold, geometric features
                    base_features = [120, 120, 1.0, 100, 60, 10, 180, 120, 80, 160, 0.35, 6, 300, 200, 0.12] + [0.1] * 10 + [55, 40, 120, 80, 85, 0.4, 0.3, 2.8, 0.7, 0.6]
                
                # Add noise to create variation
                noise = np.random.normal(0, 0.1, len(base_features))
                features = np.array(base_features) + noise
                features = np.abs(features)  # Ensure positive values
                
                synthetic_data.append(features[:35])  # Ensure exactly 35 features
                synthetic_labels.append(class_name)
        
        # Train initial model
        X = np.array(synthetic_data)
        y = np.array(synthetic_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        # Save initial model to S3
        self._save_model_to_s3()
        logger.info("Initial model created and saved to S3")
    
    def predict(self, image: np.ndarray, image_id: str = None) -> Tuple[str, float, str]:
        """Predict the class of a document region"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        # Generate image ID if not provided
        if image_id is None:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            image_id = f"img_{image_hash[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract features
        features = self._extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        # Store prediction in DynamoDB
        self._store_prediction(image_id, image, features, prediction, confidence)
        
        return prediction, confidence, image_id
    
    def _store_prediction(self, image_id: str, image: np.ndarray, features: np.ndarray, 
                         prediction: str, confidence: float):
        """Store prediction in DynamoDB"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            timestamp = datetime.now().isoformat()
            
            self.predictions_table.put_item(
                Item={
                    'image_id': image_id,
                    'image_hash': image_hash,
                    'predicted_class': prediction,
                    'confidence': Decimal(str(confidence)),
                    'timestamp': timestamp,
                    'features': features.tolist()
                }
            )
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def add_feedback(self, image_id: str, predicted_class: str, correct_class: str, confidence: float):
        """Add feedback for continual learning"""
        try:
            timestamp = datetime.now().isoformat()
            feedback_id = f"{image_id}_{timestamp}"
            
            # Store feedback
            self.feedback_table.put_item(
                Item={
                    'feedback_id': feedback_id,
                    'image_id': image_id,
                    'predicted_class': predicted_class,
                    'correct_class': correct_class,
                    'confidence': Decimal(str(confidence)),
                    'timestamp': timestamp
                }
            )
            
            # Get original prediction data and add to training data
            try:
                response = self.predictions_table.get_item(Key={'image_id': image_id})
                if 'Item' in response:
                    item = response['Item']
                    self.training_data_table.put_item(
                        Item={
                            'image_hash': item['image_hash'],
                            'features': item['features'],
                            'label': correct_class,
                            'source': 'feedback',
                            'timestamp': timestamp
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to add to training data: {e}")
                
            logger.info(f"Feedback added for {image_id}: {predicted_class} -> {correct_class}")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    def retrain_model(self, min_feedback_samples: int = 10):
        """Retrain model using accumulated feedback"""
        try:
            # Get training data from DynamoDB
            response = self.training_data_table.scan()
            items = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.training_data_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response['Items'])
            
            if len(items) < min_feedback_samples:
                logger.info(f"Not enough feedback samples for retraining ({len(items)} < {min_feedback_samples})")
                return False
            
            logger.info(f"Retraining model with {len(items)} samples...")
            
            # Prepare training data
            X = np.array([item['features'] for item in items])
            y = np.array([item['label'] for item in items])
            
            # Retrain
            X_scaled = self.scaler.fit_transform(X)
            
            self.classifier = RandomForestClassifier(
                n_estimators=min(200, max(50, len(X) // 2)),
                max_depth=12,
                min_samples_split=max(2, len(X) // 50),
                min_samples_leaf=max(1, len(X) // 100),
                random_state=42,
                class_weight='balanced'
            )
            
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                self.classifier.fit(X_train, y_train)
                y_pred = self.classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Retrained model accuracy: {accuracy:.3f}")
            else:
                self.classifier.fit(X_scaled, y)
            
            # Save updated model
            self._save_model_to_s3()
            logger.info("Model retraining completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get classifier statistics"""
        try:
            stats = {
                "total_predictions": 0,
                "total_feedback": 0,
                "accuracy": 0,
                "class_distribution": {},
                "is_trained": self.is_trained,
                "model_classes": self.classes
            }
            
            # Get prediction stats
            try:
                response = self.predictions_table.scan()
                predictions = response['Items']
                while 'LastEvaluatedKey' in response:
                    response = self.predictions_table.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    predictions.extend(response['Items'])
                
                stats["total_predictions"] = len(predictions)
                
                # Class distribution
                class_counts = {}
                for pred in predictions:
                    predicted_class = pred.get('predicted_class', 'unknown')
                    class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                stats["class_distribution"] = class_counts
                
            except Exception as e:
                logger.error(f"Failed to get prediction stats: {e}")
            
            # Get feedback stats
            try:
                response = self.feedback_table.scan()
                feedback_items = response['Items']
                while 'LastEvaluatedKey' in response:
                    response = self.feedback_table.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    feedback_items.extend(response['Items'])
                
                stats["total_feedback"] = len(feedback_items)
                
                if feedback_items:
                    correct_predictions = sum(
                        1 for item in feedback_items 
                        if item.get('predicted_class') == item.get('correct_class')
                    )
                    stats["accuracy"] = correct_predictions / len(feedback_items)
                    
            except Exception as e:
                logger.error(f"Failed to get feedback stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# Global classifier instance (reused across Lambda invocations)
classifier = None

def get_classifier():
    """Get or create classifier instance"""
    global classifier
    if classifier is None:
        classifier = DocumentRegionClassifier()
    return classifier

def process_image_data(image_data: str) -> np.ndarray:
    """Convert base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR if needed (OpenCV format)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Get HTTP method and path
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        # Initialize response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        }
        
        # Handle CORS preflight
        if http_method == 'OPTIONS':
            return response
        
        # Get classifier instance
        clf = get_classifier()
        
        # Route requests
        if path == '/health' and http_method == 'GET':
            response['body'] = json.dumps({
                'status': 'healthy',
                'model_trained': clf.is_trained,
                'timestamp': datetime.now().isoformat()
            })
            
        elif path == '/stats' and http_method == 'GET':
            stats = clf.get_stats()
            response['body'] = json.dumps(stats, default=str)
            
        elif path == '/classify' and http_method == 'POST':
            # Parse request body
            if event.get('isBase64Encoded'):
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event.get('body', '{}')
            
            request_data = json.loads(body)
            
            # Validate required fields
            if 'image_data' not in request_data:
                response['statusCode'] = 400
                response['body'] = json.dumps({'error': 'image_data is required'})
                return response
            
            # Process image and classify
            image = process_image_data(request_data['image_data'])
            image_id = request_data.get('image_id')
            
            prediction, confidence, final_image_id = clf.predict(image, image_id)
            
            response['body'] = json.dumps({
                'prediction': prediction,
                'confidence': confidence,
                'image_id': final_image_id,
                'timestamp': datetime.now().isoformat()
            })
            
        elif path == '/feedback' and http_method == 'POST':
            # Parse request body
            if event.get('isBase64Encoded'):
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event.get('body', '{}')
            
            request_data = json.loads(body)
            
            # Validate required fields
            required_fields = ['image_id', 'predicted_class', 'correct_class', 'confidence']
            for field in required_fields:
                if field not in request_data:
                    response['statusCode'] = 400
                    response['body'] = json.dumps({'error': f'{field} is required'})
                    return response
            
            # Validate correct_class
            if request_data['correct_class'] not in clf.classes:
                response['statusCode'] = 400
                response['body'] = json.dumps({
                    'error': f'Invalid class. Must be one of: {clf.classes}'
                })
                return response
            
            # Add feedback
            clf.add_feedback(
                request_data['image_id'],
                request_data['predicted_class'],
                request_data['correct_class'],
                request_data['confidence']
            )
            
            response['body'] = json.dumps({
                'message': 'Feedback received successfully',
                'image_id': request_data['image_id']
            })
            
        elif path == '/retrain' and http_method == 'POST':
            success = clf.retrain_model(min_feedback_samples=5)
            
            if success:
                message = 'Model retrained successfully'
            else:
                message = 'Not enough feedback data for retraining'
            
            response['body'] = json.dumps({'message': message, 'success': success})
            
        else:
            response['statusCode'] = 404
            response['body'] = json.dumps({'error': 'Not found'})
        
        return response
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

# For local testing
if __name__ == "__main__":
    # Test event for local development
    test_event = {
        'httpMethod': 'GET',
        'path': '/health',
        'headers': {},
        'body': None
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))