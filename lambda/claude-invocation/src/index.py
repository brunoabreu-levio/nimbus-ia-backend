import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "anthropic.claude-3-haiku-20240307-v1:0"


def lambda_handler(event, context):
    logger.info("Request: %s", event)
    response_code = 200

    try:
        body = json.loads(event['body'])
        source_code = body.get('source_code', 'No source code provided')
        prompt = body.get('prompt', 'No prompt provided')

        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.5,
            "system": source_code,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

        request = json.dumps(native_request)

        model_response = client.invoke_model(modelId=model_id, body=request)
        model_response_body = json.loads(model_response["body"].read())
        response_text = model_response_body["content"][0]["text"]

        response = {
            "statusCode": response_code,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"response": response_text}),
        }

    except Exception as e:
        logger.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        response = {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"error": str(e)}),
        }

    logger.info("Response: %s", response)
    return response
