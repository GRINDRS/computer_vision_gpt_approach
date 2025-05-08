import openai
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def resize_and_encode_image(image_path, max_size=512):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))  
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def describe_image(encoded_image):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def match_with_gpt(description):
    prompt = f"""
You are given an image description:

\"\"\"{description}\"\"\"

Which of the following artworks does it best match?

1. Starry Night  
2. Egyptian Style Statue  
3. Toy Dog  
4. Sunflowers (Van Gogh)  
5. Liberty Leading the People  
6. Mona Lisa  
7. The Scream

Respond with only the matching artwork name (e.g., "Mona Lisa") or "None" if it does not match any.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    img_path = "/Users/bhavyamalik/Desktop/egypitan_style_statue.jpg" 

    encoded = resize_and_encode_image(img_path)
    description = describe_image(encoded)
    match = match_with_gpt(description)

    if match.lower() != "none":
        print(match)
