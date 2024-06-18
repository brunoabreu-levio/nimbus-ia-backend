import boto3
import json
import logging
from multipart import parse_form_data
from io import BytesIO

REGION_NAME = "ca-central-1"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_PROMPT = "No prompt provided"
DEFAULT_SOURCE_CODE = "No source code provided"
HTTP_STATUS_OK = 200
HTTP_STATUS_INTERNAL_ERROR = 500
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_MULTIPART = "multipart/form-data"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = boto3.client("bedrock-runtime", region_name=REGION_NAME)


def parse_multipart_data(event):
    body_bytes = BytesIO(bytes(event["body"], "utf-8"))
    headers = {k.lower(): v for k, v in event['headers'].items()}
    environ = {
        'CONTENT_LENGTH': len(event['body']),
        'CONTENT_TYPE': headers.get('content-type'),
        'REQUEST_METHOD': 'POST',
        'wsgi.input': body_bytes
    }
    return parse_form_data(environ)


def handle_request_data(event):
    headers = {k.lower(): v for k, v in event['headers'].items()}
    content_type = headers.get('content-type')
    if content_type and CONTENT_TYPE_MULTIPART in content_type:
        form, files = parse_multipart_data(event)
        source_code = files.get('file', {'value': DEFAULT_SOURCE_CODE}).value
        prompt = form.get('prompt', DEFAULT_PROMPT)
    else:
        body = json.loads(event['body'])
        source_code = body.get('source_code', DEFAULT_SOURCE_CODE)
        prompt = body.get('prompt', DEFAULT_PROMPT)
    return source_code, prompt


def invoke_model(source_code, prompt):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.5,
        "system": source_code,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    model_response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body))
    return json.loads(model_response["body"].read())


def lambda_handler(event, context):
    logger.info("Request: %s", event)
    try:
        source_code, prompt = handle_request_data(event)
        model_response_body = invoke_model(source_code, prompt)
        response_text = model_response_body["content"][0]["text"]
        response = {
            "statusCode": HTTP_STATUS_OK,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"response": response_text}),
        }
    except Exception as e:
        logger.error(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        response = {
            "statusCode": HTTP_STATUS_INTERNAL_ERROR,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
    logger.info("Response: %s", response)
    return response
