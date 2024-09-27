import gradio as gr
from PIL import Image
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load models
def load_models():
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                            trust_remote_code=True, torch_dtype=torch.float32)  # float32 for CPU
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return RAG, model, processor

RAG, model, processor = load_models()
# Function for OCR and search
def ocr_and_search(image, keyword):
    text_query = "Extract all the text in Hindi and English from the image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cpu")

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2000)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        # Decode output while avoiding any coordinate information
        extracted_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        extracted_text = extracted_text.replace("The text in the image is:", "").strip()
        # Filter out any unwanted text (like coordinates)
        extracted_text = ' '.join(filter(lambda x: not any(char.isdigit() for char in x), extracted_text.split()))

    # Separate English and Hindi text using a simple heuristic
    english_text = ' '.join(filter(lambda x: all((char.islower() or char.isupper()) or char == "'"  for char in x), extracted_text.split()))
    hindi_text = ' '.join(filter(lambda x: any(ord(char) >= 128 for char in x), extracted_text.split()))

    # Perform keyword search
    keyword_lower = keyword.lower().strip()
    matched_keywords = []
    if keyword_lower:
        if keyword_lower in extracted_text.lower():
            matched_keywords = [keyword]

    # Prepare plain text output
    plain_text_output = (
        f"- English: {' '.join(english_text.split()) if english_text else 'No English text found.'}\n\n"
        f"- Hindi: {' '.join(hindi_text.split()) if hindi_text else 'No Hindi text found.'}"
    )

    return extracted_text, matched_keywords, plain_text_output

# Gradio App function
def app(image, keyword):
    # Call OCR and search function
    extracted_text, matched_keywords, plain_text_output = ocr_and_search(image, keyword)

    # Format search results
    search_results_str = "\n".join(matched_keywords) if matched_keywords else "No matches found for the keyword."

    return extracted_text, search_results_str, plain_text_output

# Gradio Interface
iface = gr.Interface(
    fn=app,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Textbox(label="Enter keyword to search in extracted text", placeholder="Keyword")
    ],
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.Textbox(label="Search Results"),
        gr.Textbox(label="Plain Text Output", lines=10)  # For plain text output
    ],
    title="Optical Character Recognition with Keyword Search from Images",
)

# Launch Gradio App
iface.launch()
