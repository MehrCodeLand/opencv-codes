from glasses_detector import GlassesClassifier

# Initialize the classifier
classifier = GlassesClassifier(kind="sunglasses")

# Directory containing images
input_dir = "./"

# Process the directory
results = classifier.process_dir(
    input_path=input_dir,
    format="proba",  # Outputs probability scores for sunglasses
    pbar=None,       # Disable the progress bar
)

# Print results
print("Recognition Results:")
for img_name, prediction in results.items():
    print(f"{img_name}: {prediction:.2f}")