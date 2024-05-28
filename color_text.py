from PIL import Image, ImageDraw, ImageFont
import random

class HighlightedTextImage:
    def __init__(self, text_tuples, font_size=48, width=1600, line_spacing=20):
        self.text_tuples = text_tuples
        self.font_size = font_size
        self.width = width
        self.line_spacing = line_spacing
        self.font = ImageFont.truetype("/home/paperspace/llama-repo/Arial.ttf", self.font_size)
        self.image = None
        self.max_char_height = self.calculate_max_char_height()
        self.height = self.calculate_height()
    
    def calculate_max_char_height(self):
        dummy_image = Image.new('RGB', (self.width, 1), (255, 255, 255))
        dummy_draw = ImageDraw.Draw(dummy_image)
        max_char_height = 0
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            bbox = dummy_draw.textbbox((0, 0), char, font=self.font)
            char_height = bbox[3] - bbox[1]
            max_char_height = max(max_char_height, char_height)
        return max_char_height
        
    def calculate_height(self):
        # Calculate height needed for the image
        dummy_image = Image.new('RGB', (self.width, 1), (255, 255, 255))
        dummy_draw = ImageDraw.Draw(dummy_image)
        
        x, y = 20, 20
        max_height = y
        max_word_height = 0

        for word, color in self.text_tuples:
            bbox = dummy_draw.textbbox((x, y), word, font=self.font)
            word_width = bbox[2] - bbox[0]
            # word_height = bbox[3] - bbox[1]
            word_height = self.max_char_height

            max_word_height = max(max_word_height, word_height)
            
            x += word_width
            if x > self.width - 40:
                x = 20
                y += max_word_height + self.line_spacing
                max_word_height = word_height
                max_height = y + max_word_height
        
        return max_height + 20  # Add some padding at the bottom

    def generate_image(self):
        self.image = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(self.image)
        
        x, y = 20, 20
        max_word_height = 0

        for word, color in self.text_tuples:
            bbox = draw.textbbox((x, y), word, font=self.font)
            word_width = bbox[2] - bbox[0]
            # word_height = bbox[3] - bbox[1]
            word_height = self.max_char_height

            max_word_height = max(max_word_height, word_height)
            
            draw.rectangle([x, y, x + word_width, y + word_height], fill=color)
            draw.text((x, y), word, fill=(0, 0, 0), font=self.font)
            
            x += word_width
            if x > self.width - 40:
                x = 20
                y += max_word_height + self.line_spacing
                max_word_height = word_height

    def save_image(self, path='highlighted_text.png', quality=95):
        if self.image:
            self.image.save(path, quality=quality)
        else:
            raise ValueError("Image not generated yet. Call generate_image() first.")

            


# # Example usage:
# text_tuples = [
#     ("In ", (255, 0, 0)),
#     ("python, ", (0, 255, 0)),
#     ("I'd ", (0, 0, 255)),
#     ("like ", (255, 255, 0)),
#     ("to ", (0, 255, 255)),
#     ("generate ", (255, 0, 255)),
#     ("an ", (128, 0, 128)),
#     ("image ", (128, 128, 0)),
#     ("of ", (0, 128, 128)),
#     ("a ", (128, 128, 128)),
#     ("paragraph ", (64, 64, 64)),
#     ("of ", (192, 192, 192)),
#     ("text ", (64, 0, 64)),
#     ("but ", (0, 64, 64)),
#     ("each ", (64, 64, 0)),
#     ("word ", (192, 0, 192)),
#     ("has ", (0, 192, 192)),
#     ("a ", (192, 192, 0)),
#     ("different ", (0, 0, 64)),
#     ("background ", (64, 0, 0)),
#     ("highlight ", (0, 64, 0)),
#     ("colour.", (64, 0, 64)),
# ]

# text_tuples = text_tuples*5

# highlighted_text_image = HighlightedTextImage(text_tuples, font_size=48, width=1600)
# highlighted_text_image.generate_image()
# highlighted_text_image.save_image('highlighted_text.png')
