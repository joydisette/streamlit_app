import base64
from pathlib import Path

def get_base64_encoded_image(image_path):
    """Get base64 encoded image from file path"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def img_to_html(image_path, alt_text=""):
    """Convert image to HTML img tag with base64 encoding"""
    try:
        img_html = f'<img src="data:image/png;base64,{get_base64_encoded_image(image_path)}" alt="{alt_text}"/>'
        return img_html
    except Exception as e:
        return f"<!-- Error loading image: {e} -->" 


from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine
from google import genai
from google.genai import types

def provide_response(project_id, model_location, model_id, augmented_rag_response, messages, temperature, datastore_id, datastore_location):

    client = genai.Client(
        vertexai=True, project=project_id, location=model_location
    )
    
    # Construct the full datastore path
    datastore_path = f"projects/{project_id}/locations/{datastore_location}/collections/default_collection/dataStores/{datastore_id}"

    # Convert messages to the format expected by the model
    contents = []
    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})

    print(contents)

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=augmented_rag_response,
            temperature=temperature,
            tools=[{"retrieval": {"vertex_ai_search": {"datastore": datastore_path}}}]
        )
    )
    
    return response.text

augmented_rag_response = """
You are a helpful assistant for the Government and Public Employees Retirement Plan. Use the retrieval search results to ground your response. 

Only use information from these results and do not make up additional information.

Please provide a natural, conversational response to the user's question while staying factual and grounded in the search results provided above.

Also make sure to include where the information is coming from in the response (url of the source).
"""