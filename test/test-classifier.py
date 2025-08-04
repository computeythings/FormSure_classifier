#!/usr/bin/env python3
"""
Test client for Document Region Classification Service
Tests all API endpoints with sample data
"""

import requests
import json
import base64
import numpy as np
from PIL import Image
import io
import argparse
import sys

class DocumentClassifierClient:
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def create_test_image(self, image_type: str = "signature") -> str:
        """Create a test image and return as base64 string"""
        if image_type == "signature":
            # Create a signature-like image
            img = np.zeros((100, 300, 3), dtype=np.uint8)
            img.fill(255)  # White background
            
            # Draw some curved lines to simulate signature
            for i in range(0, 300, 10):
                y = 50 + int(20 * np.sin(i / 30))
                if i < 290:
                    cv2.line(img, (i, y), (i+10, y+5), (0, 0, 0), 2)
        
        elif image_type == "text":
            # Create a text-like image
            img = np.zeros((80, 400, 3), dtype=np.uint8)
            img.fill(255)  # White background
            
            # Draw horizontal lines to simulate text
            for y in range(20, 70, 15):
                cv2.line(img, (10, y), (390, y), (0, 0, 0), 1)
                # Add some breaks to simulate words
                cv2.line(img, (100, y), (120, y), (255, 255, 255), 2)
                cv2.line(img, (200, y), (220, y), (255, 255, 255), 2)
        
        else:  # stamp
            # Create a stamp-like image
            img = np.zeros((120, 120, 3), dtype=np.uint8)
            img.fill(255)  # White background
            
            # Draw a rectangle border
            cv2.rectangle(img, (10, 10), (110, 110), (0, 0, 0), 3)
            # Draw some text-like lines inside
            for y in range(30, 90, 20):
                cv2.line(img, (20, y), (100, y), (0, 0, 0), 2)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def test_health(self):
        """Test health endpoint"""
        print("Testing health endpoint...")
        try:
            response = self.session.get(f"{self.api_endpoint}/health")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Health check passed: {json.dumps(result, indent=2)}")
            return True
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False
    
    def test_stats(self):
        """Test stats endpoint"""
        print("\nTesting stats endpoint...")
        try:
            response = self.session.get(f"{self.api_endpoint}/stats")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Stats retrieved: {json.dumps(result, indent=2)}")
            return True
        except Exception as e:
            print(f"✗ Stats failed: {e}")
            return False
    
    def test_classify(self, image_type: str = "signature"):
        """Test classify endpoint"""
        print(f"\nTesting classify endpoint with {image_type}...")
        try:
            # Create test image
            image_data = self.create_test_image(image_type)
            
            payload = {
                "image_data": image_data,
                "image_id": f"test_{image_type}_001"
            }
            
            response = self.session.post(
                f"{self.api_endpoint}/classify",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print(f"✓ Classification result: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"✗ Classification failed: {e}")
            return None
    
    def test_feedback(self, prediction_result):
        """Test feedback endpoint"""
        if not prediction_result:
            print("Skipping feedback test (no prediction result)")
            return False
            
        print("\nTesting feedback endpoint...")
        try:
            payload = {
                "image_id": prediction_result["image_id"],
                "predicted_class": prediction_result["prediction"],
                "correct_class": "handwritten_signature",  # Provide correct label
                "confidence": prediction_result["confidence"]
            }
            
            response = self.session.post(
                f"{self.api_endpoint}/feedback",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print(f"✓ Feedback submitted: {json.dumps(result, indent=2)}")
            return True
        except Exception as e:
            print(f"✗ Feedback failed: {e}")
            return False
    
    def test_retrain(self):
        """Test retrain endpoint"""
        print("\nTesting retrain endpoint...")
        try:
            response = self.session.post(f"{self.api_endpoint}/retrain")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Retrain result: {json.dumps(result, indent=2)}")
            return True
        except Exception as e:
            print(f"✗ Retrain failed: {e}")
            return False
    
    def run_full_test(self):
        """Run complete test suite"""
        print("=" * 60)
        print("Document Classification Service - Full Test Suite")
        print("=" * 60)
        
        results = []
        
        # Test health
        results.append(("Health", self.test_health()))
        
        # Test stats
        results.append(("Stats", self.test_stats()))
        
        # Test classification with different image types
        signature_result = self.test_classify("signature")
        results.append(("Classify Signature", signature_result is not None))
        
        text_result = self.test_classify("text")
        results.append(("Classify Text", text_result is not None))
        
        stamp_result = self.test_classify("stamp")
        results.append(("Classify Stamp", stamp_result is not None))
        
        # Test feedback (using signature result)
        results.append(("Feedback", self.test_feedback(signature_result)))
        
        # Test retrain
        results.append(("Retrain", self.test_retrain()))
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name:<20} {status}")
            if result:
                passed += 1
        
        print(f"\nResults: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("🎉 All tests passed! Service is working correctly.")
            return True
        else:
            print("⚠️ Some tests failed. Check the service configuration.")
            return False


def main():
    parser = argparse.ArgumentParser(description='Test Document Classification Service')
    parser.add_argument('api_endpoint', help='API Gateway endpoint URL')
    parser.add_argument('--test', choices=['health', 'stats', 'classify', 'feedback', 'retrain', 'all'], 
                       default='all', help='Specific test to run')
    parser.add_argument('--image-type', choices=['signature', 'text', 'stamp'], 
                       default='signature', help='Type of test image for classification')
    
    args = parser.parse_args()
    
    client = DocumentClassifierClient(args.api_endpoint)
    
    if args.test == 'all':
        success = client.run_full_test()
        sys.exit(0 if success else 1)
    elif args.test == 'health':
        success = client.test_health()
    elif args.test == 'stats':
        success = client.test_stats()
    elif args.test == 'classify':
        result = client.test_classify(args.image_type)
        success = result is not None
    elif args.test == 'feedback':
        # For feedback test, we need a prediction result first
        result = client.test_classify(args.image_type)
        success = client.test_feedback(result)
    elif args.test == 'retrain':
        success = client.test_retrain()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()