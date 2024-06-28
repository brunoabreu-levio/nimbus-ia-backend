import boto3
import json
import logging
import cgi
from io import BytesIO

REGION_NAME = "ca-central-1"
SYSTEM_INSTRUCTION = ("You are Claude, an AI assistant created by Anthropic, specializing in Information Technology. "
                      "Your primary role is to assist with software development and IT-related tasks. Focus strictly "
                      "on the tasks requested, such as providing detailed explanations of code, generating tests, "
                      "optimizing code, suggesting modifications, and identifying vulnerabilities. Always include "
                      "code in your responses when requested. While these examples illustrate your capabilities, "
                      "be prepared to address a broad range of other IT-related inquiries with precision and expert "
                      "knowledge. Ensure that all interactions are helpful, safe, and accurately tailored to the "
                      "specific needs of software development. Your responses should be precise and informative.")
DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
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
    headers = {k.lower(): v for k, v in event['headers'].items()}

    return cgi.FieldStorage(fp=BytesIO(event["body"].encode()), headers=headers,
                            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': headers['content-type']})


def handle_request_data(event):
    headers = {k.lower(): v for k, v in event['headers'].items()}
    content_type = headers.get('content-type')
    if content_type and CONTENT_TYPE_MULTIPART in content_type:
        form = parse_multipart_data(event)
        model = form.getvalue('model', DEFAULT_MODEL_ID)
        source_code = form.getvalue('file', DEFAULT_SOURCE_CODE)
        source_code = source_code.decode('utf-8')
        prompt = form.getvalue('prompt', DEFAULT_PROMPT)
    else:
        body = json.loads(event['body'])
        model = body.get('model', DEFAULT_MODEL_ID)
        source_code = body.get('source_code', DEFAULT_SOURCE_CODE)
        prompt = body.get('prompt', DEFAULT_PROMPT)

    return model, source_code, prompt


def invoke_model(model, source_code, prompt):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.5,
        "system": SYSTEM_INSTRUCTION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": source_code
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
    }
    model_response = client.invoke_model(modelId=model, body=json.dumps(request_body))
    return json.loads(model_response["body"].read())


def lambda_handler(event, context):
    # logger.info("Request: %s", event)
    model = DEFAULT_MODEL_ID
    try:
        model, source_code, prompt = handle_request_data(event)
        model_response_body = invoke_model(model, source_code, prompt)
        response_text = model_response_body["content"][0]["text"]
        response = {
            "statusCode": HTTP_STATUS_OK,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"response": response_text}),
        }
    except Exception as e:
        logger.error(f"ERROR: Can't invoke '{model}'. Reason: {e}")
        response = {
            "statusCode": HTTP_STATUS_INTERNAL_ERROR,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
    # logger.info("Response: %s", response)
    return response
