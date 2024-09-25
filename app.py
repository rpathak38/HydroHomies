import os
import base64
import torch
import random
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn as nn
import folium
from shapely.geometry import Point, Polygon

# Define the latitude and longitude boundaries of Manhattan
manhattan_boundary = Polygon([
    (-74.0107, 40.7003), (-73.9712, 40.8721), # Rough boundary points of Manhattan island
    (-73.9331, 40.8000), (-73.935242, 40.785091), # Eastern boundary points
    (-74.0169, 40.7321), (-74.0173, 40.7090), # Western and southern boundaries
    (-74.0107, 40.7003)  # Closing the polygon loop
])

# Define street names associated with randomized coordinates
street_names = [
    "Broadway & Chambers St", "E 14th St & 3rd Ave", "W 47th St & 7th Ave",
    "E 42nd St & Lexington Ave", "W 34th St & 7th Ave", "W 29th St & 11th Ave",
    "W 27th St & 10th Ave", "E 12th St & 2nd Ave", "Vesey St & West St",
    "E Houston St & Bowery", "E 48th St & Madison Ave", "W 30th St & 7th Ave",
    "W 4th St & Waverly Pl", "E 45th St & Vanderbilt Ave", "W 3rd St & Thompson St",
    "Warren St & West St", "W 50th St & 6th Ave", "E 43rd St & Madison Ave",
    "W 24th St & 9th Ave", "W 33rd St & 6th Ave"
]

# Function to generate random coordinates within the Manhattan polygon
def generate_random_coordinate_within_manhattan():
    while True:
        lat = random.uniform(40.7000, 40.8800)  # Adjusted tighter latitudes for Manhattan
        lon = random.uniform(-74.0200, -73.9300)  # Adjusted tighter longitudes for Manhattan
        point = Point(lon, lat)
        if manhattan_boundary.contains(point):
            return (lat, lon)

# Generate randomized coordinates within Manhattan for each street name
randomized_coordinates = [
    (generate_random_coordinate_within_manhattan(), street)
    for street in street_names
]

# Path to the folder containing your images
image_folder = 'images'  # Ensure the images are in the correct folder relative to this script

# Gather .jpg images from the folder
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]

# Randomize the pairing of images and coordinates
random.shuffle(image_files)
location_data = list(zip(randomized_coordinates, image_files))

# Function to load the model with weights from the .pth file
def load_model(model_path):
    # Load a pre-trained MobileNetV2 model
    model = models.mobilenet_v2(pretrained=False)  # Set pretrained=False since we're loading custom weights
    
    # Modify the classifier to match the training setup
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(1280, 1)  # Output size is 1 for binary classification (flooded or not)
    )

    # Load the weights from the .pth file
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Define image transformation to match model input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model's input size
    transforms.ToTensor(),
])

# Function to predict whether the image shows flooding (1) or not (0)
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(image)
        return (prediction > 0).item()  # Returns 1 for flooded, 0 for not flooded

# Function to label images with their flood status for visual verification
def label_image_with_prediction(image_path, prediction):
    label = "Flooded" if prediction == 1 else "Not Flooded"
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), label, fill="red" if prediction == 1 else "green", font=font)
    labeled_image_path = f"labeled_{os.path.basename(image_path)}"
    image.save(labeled_image_path)
    return labeled_image_path

# Load your model - update with the path to your .pth file
model = load_model('C:\\Users\\kvenkate3\\Desktop\\run_1_model.pth')  # Replace this with the correct path to your .pth file

# Predict flood status for each image and ensure correct association
results = []
for (coord, street_name), img_path in location_data:
    flood_status = predict_image(model, img_path)
    labeled_image_path = label_image_with_prediction(img_path, flood_status)
    
    # Debugging: print the associated data
    print(f"Processed {img_path} -> Prediction: {'Flooded' if flood_status else 'Not Flooded'} at {street_name} ({coord})")
    
    results.append(((coord, street_name), flood_status, labeled_image_path))

# Extract coordinates to fit map bounds automatically
map_coordinates = [coord[0] for coord, _, _ in results]

# Create a Folium map with bounds set to fit all the pins
m = folium.Map(location=[40.7831, -73.9712], zoom_start=12, width='100%', height='600px')
m.fit_bounds([min(map_coordinates), max(map_coordinates)])

# Function to get marker color based on prediction
def get_marker_color(prediction):
    return 'red' if prediction == 1 else 'green'  # Red for flooded, green for not flooded

# Function to encode image in base64 for embedding
def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Add markers to the map based on prediction results
for (coord, street_name), pred, img_path in results:
    # Encode the image to base64 to embed directly in HTML
    encoded_image = encode_image_base64(img_path)
    # Combine the street name, image, and flooded status into a single tooltip
    tooltip_html = f'''
    <b>{street_name}</b><br>
    <img src="data:image/jpeg;base64,{encoded_image}" width="200"><br>
    Flooded: {"Yes" if pred == 1 else "No"}
    '''
    folium.Marker(
        location=coord,
        popup=folium.Popup(html=tooltip_html, max_width=300),
        icon=folium.Icon(color=get_marker_color(pred))
    ).add_to(m)

# Sort the results so flooded locations appear at the top
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# Create an HTML list for the ranking bar on the right-hand side
ranking_html = '<div class="ranking-container"><h4>Flood Ranking</h4><ul>'
for idx, ((_, street_name), pred, _) in enumerate(sorted_results):
    status = "Flooded" if pred == 1 else "Not Flooded"
    ranking_html += f'<li>{idx + 1}. Location: {street_name}, Status: {status}</li>'
ranking_html += '</ul></div>'

# Add the ranking bar and title to the map layout using Folium HTML elements
layout_html = f'''
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div style="flex: 3; border: 2px solid black; padding: 10px; margin-right: 10px;">
            <h3 style="text-align: center; font-size: 20px;"><b>NYC Real Time Stormwater Management Map</b></h3>
            <div style="width: 100%;">{m._repr_html_()}</div> <!-- Ensure the map fills the entire width -->
        </div>
        <div style="flex: 1; height: 600px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
            {ranking_html}
        </div>
    </div>
'''

# Save the layout to an HTML file manually
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Real Time Stormwater Management Map</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        h3 {{
            margin: 0;
        }}
        .ranking-container {{
            font-size: 14px;
        }}
        ul {{
            padding: 0;
            list-style-type: none;
        }}
        li {{
            padding: 5px 0;
        }}
    </style>
</head>
<body>
    {layout_html}
</body>
</html>
"""

# Save the HTML output to a file
with open('map_with_images.html', 'w') as file:
    file.write(html_output)

print("Map has been saved as map_with_images.html")