import random
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine tuned model and processor
processor_trained = BlipProcessor.from_pretrained("saarah005/blip-finetuned-holmes")
model_trained = BlipForConditionalGeneration.from_pretrained("saarah005/blip-finetuned-holmes").to(device)


def holmesify(caption):
    # Small dictionary of style replacements
    replacements = {
    # ---- People ----
    "dog": "hound",
    "dogs": "hounds",
    "cat": "feline companion",
    "cats": "feline companions",
    "bird": "avian companion",
    "birds": "avian companions",
    "man": "gentleman",
    "men": "gentlemen",
    "woman": "lady",
    "women": "ladies",
    "boys": "lads",
    "boy": "lad",
    "girl": "maiden",
    "girls": "maids",
    "child": "youth",
    "children": "young ones",
    "person": "individual of interest",
    "people": "various passersby",
    "crowd": "assembly of onlookers",

    # ---- Actions / Verbs ----
    "running": "in swift pursuit",
    "walking": "taking measured strides",
    "standing": "stationed with deliberate poise",
    "sitting": "resting in quiet contemplation",
    "talking": "engaged in hushed discourse",
    "laughing": "expressing a moment of levity",
    "smiling": "bearing a subtle expression of warmth",
    "looking": "fixing one's gaze with intent",
    "pointing": "gesturing with purposeful direction",
    "holding": "grasping with notable care",
    "playing": "engaged in curious amusement",
    "jumping": "springing forth with unexpected vigor",
    "working": "occupied with some industrious task",

    # ---- Objects ----
    "car": "motorised carriage",
    "bike": "mechanical velocipede",
    "bicycle": "two-wheeled contraption",
    "ball": "spherical object of diversion",
    "bag": "carried satchel",
    "phone": "telephonic device",
    "camera": "photographic apparatus",
    "book": "well-worn volume",
    "hat": "notable headwear",

    # ---- Locations / Environments ----
    "beach": "windswept shore",
    "street": "fog-laden lane",
    "road": "lonely thoroughfare",
    "park": "quiet public green",
    "forest": "shadowed woodland",
    "city": "bustling metropolis",
    "house": "residence of uncertain history",
    "room": "chamber of modest proportion",
    "yard": "narrow courtyard",
    "river": "meandering waterway",

    # ---- Atmosphere / Adjectives ----
    "beautiful": "most striking in its appearance",
    "dark": "shrouded in somber gloom",
    "bright": "lit with uncommon clarity",
    "large": "of considerable magnitude",
    "small": "modest in scale",
    "happy": "in unusually pleasant spirits",
    "old": "weathered by time",
    "new": "freshly appointed",

    # ---- Extra stylistic swaps ----
    "near": "in the immediate vicinity of",
    "beside": "adjacent to",
    "behind": "situated just beyond",
    "before": "presented directly before",
    "with": "accompanied by",
    }
    words = caption.split()
    new_words = [
    replacements.get(w.lower().strip(",."), w.strip(",."))
    for w in words
    ]
    stylized = " ".join(new_words)

    # Add a Holmesian flourish
    intros = [
        "Upon my keen observation,",
        "It was immediately apparent to my trained eye that",
        "After a brief yet thorough inspection, I deduced that",
        "From even the faintest clues, one might surmise",
        "To the ordinary passerby it may seem trivial, yet I perceived that",
        ]
    outros = [
        " — a detail insignificant to most, yet crucial to the discerning mind.",
        " — a sight which, though mundane, whispered of deeper implications.",
        " — revealing a subtle narrative hidden beneath the everyday scene.",
        " — an occurrence that beckons further inquiry to the vigilant observer.",

    ]
    holmes_caption = f"{random.choice(intros)} {stylized}{random.choice(outros)}"
    return holmes_caption

def infer(image):
        # Preprocess
        inputs = processor_trained(images=image, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]

        # Generate BLIP caption
        model_trained.eval()
        with torch.no_grad():
            generated_ids = model_trained.generate(pixel_values=pixel_values, max_length=50)
        caption = processor_trained.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # To Holmesify
        return holmesify(caption)


custom_css = """

/* Host container holds all theme variables */
:host {
    --bg: #0d0d0d;
    --text: #e4e4e4;
    --border-color: #444;
}

/* Ensure background covers entire screen */
.wrapper {
    min-height: 100vh !important;
    width: 100vw !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    padding: 20px;
}

/* Header container */
.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
}

/* Circular portrait image */
.header-img {
    width: 70px;
    aspect-ratio: 1 / 1;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid white;
}

/* Sherlock title */
.main-heading {
    font-family: "Great Vibes", cursive;
    font-size: 42px;
    font-weight: bold;
    margin: 0;
    color: var(--text);
    text-align: center;
}

/* Equal-height layout fix */
.equal-box {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.equal-box textarea {
    flex: 1 !important;
    height: 100% !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-color) !important;
}

/* Button styling */
.sherlock-btn {
    background-color: #692625 !important;
    color: white !important;
    border: 2px solid #4d1b1b !important;
    font-weight: bold;
    transition: 0.15s ease-in-out;
    width: 250px !important;
    margin: 20px auto !important;
    display: block !important;
    text-align: center !important;
}

.sherlock-btn:hover {
    background-color: #8a2f2e !important;
    transform: scale(1.03);
}

"""
def load_fonts():
    return gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">
    """)


with gr.Blocks(elem_classes=["wrapper"]) as demo:

    # Injecting CSS
    gr.HTML(f"<style>{custom_css}</style>")

    # Load external font
    load_fonts()

    # Header
    gr.HTML("""
    <div class="header-container">
        <img src="https://www.mysterycenter.com/wp-content/uploads/2024/06/dreamstime_s_50012210-640x381.jpg" class="header-img">
        <div class="main-heading">Sherlock Holmes Caption Generator</div>
    </div>
    """)

    with gr.Row(equal_height=True):
        image_in = gr.Image(
            type="pil",
            label="Upload Image",
            sources=["upload"],
            elem_classes=["equal-box"]
        )

        output_box = gr.Textbox(
            label="Sherlock Holmes Caption",
            lines=10,
            show_copy_button=True,
            elem_classes=["equal-box"]
        )

    run_btn = gr.Button("Generate Sherlock Caption", elem_classes=["sherlock-btn"])
    run_btn.click(infer, inputs=image_in, outputs=output_box)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_api=False,
    share=False,
)


