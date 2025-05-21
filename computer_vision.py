import openai
import base64
import os
import sys
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import cv2

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def resize_and_encode_image(image_path, max_size=512):
    print(f"[INFO] Loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print(f"[INFO] Image successfully encoded. Length: {len(encoded)} characters")
        return encoded
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        sys.exit(1)

def match_image_to_artwork(encoded_image, artworks):
    try:
        artwork_lines = "\n".join([
            f"{name}: {', '.join(tags)}"
            for name, tags in artworks.items()
        ])

        system_prompt = f"""You are a simple artwork classifier.

Compare the image to this list of artworks. If the image matches any artwork, return ONLY the artwork name.
If the image doesn't match any artwork, return ONLY the word "wall".

Artworks:
{artwork_lines}
"""

        print("[INFO] Sending image and matching request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=50,
        )

        result = response.choices[0].message.content.strip().lower()
        # If the result is not one of our known artworks, return "wall"
        if result not in [name.lower() for name in artworks.keys()]:
            return "wall"
        return result
    except Exception as e:
        print(f"[ERROR] OpenAI API failed: {e}")
        return "wall"

if __name__ == "__main__":
    artworks = {
        "Starry Night": {
            "starry-night", "van-gogh", "swirling-sky", "yellow-stars", "blue-sky",
            "cypress-tree", "village-at-night", "expressionist-art", "moon",
            "blue-and-yellow-painting", "famous-artwork", "post-impressionism"
        },
        "Stylized Egyptian Sculpture": {
            "metal-figurine", "brass-statue", "decorative-figure", "ethnic-art",
            "tribal-sculpture", "african-style-decor", "woman-holding-bowl",
            "red-and-black-dress", "ornamental-design", "engraved-base",
            "painted-metal-statue", "folk-art-sculpture", "bronze-body-figure"
        },
        "Toy Dog": {
            "plush_dog", "brown_dog", "toy_dog", "fabric_dog", "stuffed_animal",
            "dog_doorstop", "bead_eyes", "bow_collar", "floppy_ears", "round_body"
        },
        "Sunflowers (Van Gogh)": {
            "sunflowers", "vase with flowers", "yellow petals", "wilted petals",
            "green stems", "warm ochre background", "post-impressionist style",
            "Van Gogh signature", "textured impasto brushwork"
        },
        "Liberty Leading the People": {
            "romanticism", "oil painting", "historical painting", "Eugène Delacroix",
            "revolutionary scene", "French flag", "bare-breasted woman", "tricolour flag",
            "heroic symbolism"
        },
        "Mona Lisa": {
            "portrait", "woman", "smile", "Leonardo da Vinci", "Renaissance",
            "sfumato", "folded hands", "calm expression"
        },
        "The Scream": {
            "the scream", "edvard munch", "screaming figure", "hands on face",
            "swirling sky", "expressionist style", "vivid colours", "psychological expression"
        }
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        sys.exit(1)

    print("Press ENTER to capture a frame for analysis. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        cv2.imshow("Press ENTER to analyze frame", frame)
        key = cv2.waitKey(1)

        if key == 13:  
            filename = "frame.jpg"
            cv2.imwrite(filename, frame)
            print("[INFO] Frame captured. Analyzing...")
            try:
                encoded = resize_and_encode_image(filename)
                match = match_image_to_artwork(encoded, artworks)
                print(f"[RESULT] Matched artwork: {match}")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif key == 27:  
            print("[INFO] Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
