import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import soundfile as sf

class GenreCNN(nn.Module):
    """CNN model for music genre classification"""
    
    def __init__(self, num_classes=10):
        super(GenreCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Fourth conv block
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def predict_genre(image_path, model_path='genre_classifier_model.pth'):
    """
    Predict the genre of a music spectrogram image
    
    Args:
        image_path (str): Path to the spectrogram image
        model_path (str): Path to the trained model
    
    Returns:
        tuple: (predicted_genre, confidence_score)
    """
    # Define genre classes (in the same order as training)
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = GenreCNN(num_classes=len(genres))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_genre = genres[predicted.item()]
        confidence_score = confidence.item()
    
    return predicted_genre, confidence_score

def predict_multiple_images(image_dir, model_path='genre_classifier_model.pth'):
    """
    Predict genres for all images in a directory
    
    Args:
        image_dir (str): Directory containing spectrogram images
        model_path (str): Path to the trained model
    
    Returns:
        dict: Dictionary mapping image names to predictions
    """
    results = {}
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to classify...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            genre, confidence = predict_genre(image_path, model_path)
            results[image_file] = {
                'genre': genre,
                'confidence': confidence
            }
            print(f"{image_file}: {genre} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results[image_file] = {'genre': 'error', 'confidence': 0.0}
    
    return results

 

def audio_to_spectrogram(audio_path, output_size=(64, 64)):
    """
    Convert audio file (MP3, WAV, etc.) to spectrogram image
    
    Args:
        audio_path (str): Path to the audio file (MP3, WAV, etc.)
        output_size (tuple): Size of the output spectrogram image
    
    Returns:
        PIL.Image: Spectrogram image
    """
    try:
        # Load audio file (librosa supports MP3, WAV, and other formats)
        y, sr = librosa.load(audio_path, duration=30)  # Load first 30 seconds
        
        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize(output_size)
        plt.close()
        
        return img
        
    except Exception as e:
        print(f"Error converting audio to spectrogram: {e}")
        return None

def mp3_to_spectrogram(mp3_path, output_size=(64, 64)):
    """
    Convert MP3 file to spectrogram image (alias for audio_to_spectrogram)
    
    Args:
        mp3_path (str): Path to the MP3 file
        output_size (tuple): Size of the output spectrogram image
    
    Returns:
        PIL.Image: Spectrogram image
    """
    return audio_to_spectrogram(mp3_path, output_size)

def predict_genre_from_audio(audio_path, model_path='genre_classifier_model.pth'):
    """
    Predict the genre of an audio file (MP3, WAV, etc.) by converting it to a spectrogram
    
    Args:
        audio_path (str): Path to the audio file (MP3, WAV, etc.)
        model_path (str): Path to the trained model
    
    Returns:
        tuple: (predicted_genre, confidence_score)
    """
    try:
        # Convert audio to spectrogram
        spectrogram_img = audio_to_spectrogram(audio_path)
        
        if spectrogram_img is None:
            return None, 0.0
        
        # Save temporary image
        temp_path = 'temp_spectrogram.png'
        spectrogram_img.save(temp_path)
        
        # Predict genre using the image prediction function
        genre, confidence = predict_genre(temp_path, model_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return genre, confidence
        
    except Exception as e:
        print(f"Error predicting genre from audio: {e}")
        return None, 0.0

def predict_genre_from_mp3(mp3_path, model_path='genre_classifier_model.pth'):
    """
    Predict the genre of an MP3 file by converting it to a spectrogram (alias for predict_genre_from_audio)
    
    Args:
        mp3_path (str): Path to the MP3 file
        model_path (str): Path to the trained model
    
    Returns:
        tuple: (predicted_genre, confidence_score)
    """
    return predict_genre_from_audio(mp3_path, model_path)

def predict_multiple_mp3s(mp3_dir, model_path='genre_classifier_model.pth'):
    """
    Predict genres for all MP3 files in a directory
    
    Args:
        mp3_dir (str): Directory containing MP3 files
        model_path (str): Path to the trained model
    
    Returns:
        dict: Dictionary mapping MP3 filenames to predictions
    """
    results = {}
    
    # Get all MP3 files
    mp3_files = [f for f in os.listdir(mp3_dir) if f.lower().endswith('.mp3')]
    
    if not mp3_files:
        print("No MP3 files found in directory")
        return results
    
    print(f"Found {len(mp3_files)} MP3 files to classify...")
    
    for mp3_file in mp3_files:
        mp3_path = os.path.join(mp3_dir, mp3_file)
        try:
            genre, confidence = predict_genre_from_audio(mp3_path, model_path)
            results[mp3_file] = {
                'genre': genre,
                'confidence': confidence
            }
            print(f"{mp3_file}: {genre} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")
            results[mp3_file] = {'genre': 'error', 'confidence': 0.0}
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Music Genre Prediction Tool")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'genre_classifier_model.pth'
    if not os.path.exists(model_path):
        print("Error: Model file 'genre_classifier_model.pth' not found!")
        print("Please run 'genre_classifier.py' first to train the model.")
    else:
        # Predict genre for the WAV file
        wav_file = "/Users/xuruining/Desktop/country.00001 copy.wav"
        
        if os.path.exists(wav_file):
            print(f"\nPredicting genre for: {wav_file}")
            print("-" * 50)
            genre, confidence = predict_genre_from_audio(wav_file, model_path)
            
            if genre:
                print(f"\n✓ Predicted Genre: {genre}")
                print(f"✓ Confidence: {confidence:.1%} ({confidence:.3f})")
            else:
                print("✗ Error: Could not predict genre from audio file.")
        else:
            print(f"Error: Audio file not found at {wav_file}")
            print("\nExample usage with other files:")
            print("-" * 50)
            
            # Example: Predict a single image
            example_image = "/Users/xuruining/Desktop/genre, image/images_original/blues/blues00000.png"
            
            if os.path.exists(example_image):
                print(f"\nPredicting genre for: {example_image}")
                genre, confidence = predict_genre(example_image)
                print(f"Predicted Genre: {genre}")
                print(f"Confidence: {confidence:.3f}")
            else:
                print("Example image not found. Please provide a valid image path.")
                
            # Example: Predict all images in a directory
            # Uncomment the following lines to predict all images in a directory
            # image_directory = "/Users/xuruining/Desktop/genre, image/images_original/blues"
            # results = predict_multiple_images(image_directory)
