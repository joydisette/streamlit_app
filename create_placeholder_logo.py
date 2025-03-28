from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_logo(text, output_path, size=(200, 60), bg_color="white", text_color="black"):
    """Create a placeholder logo with text"""
    # Create new image with white background
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("Arial", 24)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate text position (center)
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

if __name__ == "__main__":
    # Create placeholder logos
    create_placeholder_logo("CN", "static/images/cn_logo.png", text_color="#CC0033")
    create_placeholder_logo("DS Group", "static/images/data_science_group_logo.png") 