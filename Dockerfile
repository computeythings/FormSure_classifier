# Multi-stage build for AWS Lambda deployment
FROM public.ecr.aws/lambda/python:3.11 as builder

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ cmake make wget unzip && \
    yum clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}/

# Production stage
FROM public.ecr.aws/lambda/python:3.11

# Copy installed dependencies
COPY --from=builder ${LAMBDA_TASK_ROOT}/ ${LAMBDA_TASK_ROOT}/

# Copy application code
COPY classifier.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["classifier.lambda_handler"]