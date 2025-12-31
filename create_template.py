from PIL import Image, ImageDraw, ImageFilter
import os

def create_gradient_template(width=1080, height=1920, filename='templates/template.png'):
    """Create a beautiful gradient template"""
    os.makedirs('templates', exist_ok=True)
    
    # Create base image
    template = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(template)
    
    # Create gradient (light blue to white)
    for y in range(height):
        # Calculate color for this row
        ratio = y / height
        r = int(240 - ratio * 30)
        g = int(248 - ratio * 20)
        b = int(255 - ratio * 10)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    template.save(filename)
    print(f"Template saved to {filename}")
    return template


def create_professional_template(width=1080, height=1920, filename='templates/template.png'):
    """Create a professional gradient template with subtle design"""
    os.makedirs('templates', exist_ok=True)
    
    template = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(template)
    
    # Gradient from top to bottom (soft blue-gray)
    for y in range(height):
        ratio = y / height
        r = int(230 + ratio * 20)
        g = int(235 + ratio * 15)
        b = int(245 + ratio * 5)
        draw.line([(0, y), (width, y)], fill=(min(r, 255), min(g, 255), min(b, 255)))
    
    # Add subtle floor shadow effect at bottom
    floor_start = int(height * 0.85)
    for y in range(floor_start, height):
        ratio = (y - floor_start) / (height - floor_start)
        overlay_alpha = int(30 * ratio)
        r = max(0, 230 - overlay_alpha)
        g = max(0, 235 - overlay_alpha)
        b = max(0, 245 - overlay_alpha)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    template.save(filename)
    print(f"Professional template saved to {filename}")
    return template


def create_colored_template(width=1080, height=1920, color='blue', filename='templates/template.png'):
    """Create template with different color themes"""
    os.makedirs('templates', exist_ok=True)
    
    template = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(template)
    
    # Color themes
    themes = {
        'blue': {'start': (200, 220, 255), 'end': (240, 248, 255)},
        'green': {'start': (200, 240, 200), 'end': (240, 255, 240)},
        'pink': {'start': (255, 220, 230), 'end': (255, 245, 250)},
        'orange': {'start': (255, 230, 200), 'end': (255, 250, 240)},
        'purple': {'start': (230, 210, 250), 'end': (248, 240, 255)},
        'gray': {'start': (200, 200, 200), 'end': (245, 245, 245)},
        'white': {'start': (255, 255, 255), 'end': (255, 255, 255)},
    }
    
    theme = themes.get(color, themes['blue'])
    start = theme['start']
    end = theme['end']
    
    for y in range(height):
        ratio = y / height
        r = int(start[0] + ratio * (end[0] - start[0]))
        g = int(start[1] + ratio * (end[1] - start[1]))
        b = int(start[2] + ratio * (end[2] - start[2]))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    template.save(filename)
    print(f"{color.capitalize()} template saved to {filename}")
    return template


def create_studio_template(width=1080, height=1920, filename='templates/template.png'):
    """Create studio-style template with vignette effect"""
    os.makedirs('templates', exist_ok=True)
    
    template = Image.new('RGB', (width, height), (240, 240, 245))
    draw = ImageDraw.Draw(template)
    
    # Create radial gradient (vignette effect)
    center_x, center_y = width // 2, height // 2
    max_dist = ((width/2)**2 + (height/2)**2) ** 0.5
    
    for y in range(height):
        for x in range(width):
            dist = ((x - center_x)**2 + (y - center_y)**2) ** 0.5
            ratio = dist / max_dist
            darkness = int(ratio * 40)
            r = max(0, 245 - darkness)
            g = max(0, 245 - darkness)
            b = max(0, 250 - darkness)
            template.putpixel((x, y), (r, g, b))
    
    # Add floor gradient
    floor_start = int(height * 0.75)
    for y in range(floor_start, height):
        ratio = (y - floor_start) / (height - floor_start)
        for x in range(width):
            current = template.getpixel((x, y))
            darkness = int(20 * ratio)
            r = max(0, current[0] - darkness)
            g = max(0, current[1] - darkness)
            b = max(0, current[2] - darkness)
            template.putpixel((x, y), (r, g, b))
    
    template.save(filename)
    print(f"Studio template saved to {filename}")
    return template


def create_solid_template(width=1080, height=1920, color=(255, 255, 255), filename='templates/template.png'):
    """Create solid color template"""
    os.makedirs('templates', exist_ok=True)
    
    template = Image.new('RGB', (width, height), color)
    template.save(filename)
    print(f"Solid template saved to {filename}")
    return template


# Create all templates
if __name__ == '__main__':
    print("Creating templates...")
    
    # Default template
    create_professional_template(filename='templates/template.png')
    
    # Additional templates
    create_gradient_template(filename='templates/gradient.png')
    create_studio_template(filename='templates/studio.png')
    create_colored_template(color='blue', filename='templates/blue.png')
    create_colored_template(color='green', filename='templates/green.png')
    create_colored_template(color='pink', filename='templates/pink.png')
    create_colored_template(color='white', filename='templates/white.png')
    create_solid_template(color=(50, 50, 50), filename='templates/dark.png')
    
    print("\nAll templates created in 'templates/' folder!")