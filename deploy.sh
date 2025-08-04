#!/bin/bash

# Deploy Document Region Classification Service to AWS Lambda
# Usage: ./deploy.sh [environment] [region] [bucket-name]

set -e

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
S3_BUCKET_BASE=${3:-document-classifier-models}
STACK_NAME="document-classifier-${ENVIRONMENT}"
TABLE_PREFIX="classifier"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Document Region Classification Service Deployment ===${NC}"
echo -e "Environment: ${GREEN}${ENVIRONMENT}${NC}"
echo -e "Region: ${GREEN}${AWS_REGION}${NC}"
echo -e "S3 Bucket Base: ${GREEN}${S3_BUCKET_BASE}${NC}"
echo

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install AWS CLI first.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="${S3_BUCKET_BASE}-${AWS_ACCOUNT_ID}-${AWS_REGION}"

echo -e "${YELLOW}Step 1: Creating S3 bucket for deployment artifacts...${NC}"
if ! aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3 mb "s3://${S3_BUCKET}"
    else
        aws s3 mb "s3://${S3_BUCKET}" --region "$AWS_REGION"
    fi
    echo -e "${GREEN}S3 bucket created: ${S3_BUCKET}${NC}"
else
    echo -e "${GREEN}S3 bucket already exists: ${S3_BUCKET}${NC}"
fi

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled

echo -e "${YELLOW}Step 2: Creating Python dependencies layer...${NC}"
mkdir -p build/layer/python
pip install -r requirements.txt -t build/layer/python/ --no-deps
cd build/layer
zip -r9 ../dependencies.zip .
cd ../..

# Upload dependencies layer
aws s3 cp build/dependencies.zip "s3://${S3_BUCKET}/layers/dependencies.zip"
echo -e "${GREEN}Dependencies layer uploaded${NC}"

echo -e "${YELLOW}Step 3: Packaging Lambda function code...${NC}"
zip -r build/classifier.zip classifier.py
aws s3 cp build/classifier.zip "s3://${S3_BUCKET}/code/classifier.zip"
echo -e "${GREEN}Function code uploaded${NC}"

echo -e "${YELLOW}Step 4: Deploying CloudFormation stack...${NC}"
aws cloudformation deploy \
    --template-file cloudformation.yaml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
        EnvironmentName="$ENVIRONMENT" \
        TablePrefix="$TABLE_PREFIX" \
        S3BucketName="$S3_BUCKET_BASE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$AWS_REGION"

echo -e "${GREEN}CloudFormation stack deployed successfully!${NC}"

echo -e "${YELLOW}Step 5: Getting deployment outputs...${NC}"
API_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
    --output text \
    --region "$AWS_REGION")

LAMBDA_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionArn`].OutputValue' \
    --output text \
    --region "$AWS_REGION")

echo -e "${BLUE}=== Deployment Complete ===${NC}"
echo -e "API Endpoint: ${GREEN}${API_ENDPOINT}${NC}"
echo -e "Lambda Function ARN: ${GREEN}${LAMBDA_ARN}${NC}"
echo -e "S3 Bucket: ${GREEN}${S3_BUCKET}${NC}"
echo

echo -e "${YELLOW}Testing the deployment...${NC}"
HEALTH_URL="${API_ENDPOINT}/health"
echo -e "Testing health endpoint: ${HEALTH_URL}"

# Wait a moment for the API to be ready
sleep 10

if curl -s -f "$HEALTH_URL" > /dev/null; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    curl -s "$HEALTH_URL" | python -m json.tool
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "The API might need a few more minutes to be fully ready."
fi

echo
echo -e "${BLUE}=== API Endpoints ===${NC}"
echo -e "Health: ${GREEN}GET ${API_ENDPOINT}/health${NC}"
echo -e "Stats: ${GREEN}GET ${API_ENDPOINT}/stats${NC}"
echo -e "Classify: ${GREEN}POST ${API_ENDPOINT}/classify${NC}"
echo -e "Feedback: ${GREEN}POST ${API_ENDPOINT}/feedback${NC}"
echo -e "Retrain: ${GREEN}POST ${API_ENDPOINT}/retrain${NC}"

echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test the classification endpoint with image data"
echo "2. Provide feedback to improve the model"
echo "3. Monitor performance in CloudWatch logs"
echo "4. Use the retrain endpoint to update the model with new data"

# Cleanup build directory
rm -rf build/
echo -e "${GREEN}Deployment complete!${NC}"