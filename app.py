import streamlit as st
from music_prediction import music_prediction
import os

st.set_page_config(
    page_title="Music Analysis Platform",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Analysis Platform")
st.markdown("Upload an audio file to analyze its genre, instruments, structure, and BPM. The accuracy is improving.")
st.markdown("Use the prompt generated below to create your own music with the same genre, instruments, structure, and BPM in AI music generation model.")
st.markdown("&nbsp;")

# External website link
st.markdown("🔗 Visit AI Music Generation Platform: [Suno—AI for Music Creators] https://www.suno.com")
st.markdown("🔗 Visit AI Music Generation Platform: [Aiva—AI for Music Creators] https://www.aiva.ai")
st.markdown("🔗 Visit AI Music Generation Platform: [Soundraw—AI for Music Creators] https://soundraw.io")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
    help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"./temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display audio player
    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    # Run analysis
    if st.button("Analyze Music", type="primary"):
        with st.spinner("Analyzing music... This may take a moment."):
            try:
                metainfo = music_prediction(temp_path)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎸 Genre")
                    genre = metainfo["genre"]["result"]
                    confidence = metainfo["genre"]["confidence"]
                    st.metric("Genre", genre, f"{float(confidence) * 100:.1f}% confidence")
                    
                    st.subheader("🎹 Instruments")
                    instruments = metainfo["instrument"]
                    for inst in instruments[:5]:  # Show top 5
                        st.write(f"• **{inst['instrument']}** ({float(inst['confidence']) * 100:.1f}%)")
                
                with col2:
                    st.subheader("📊 BPM")
                    bpm = metainfo["bpm"]["result"]
                    st.metric("Beats Per Minute", f"{bpm:.0f}")
                    
                    st.subheader("🎼 Structure")
                    structure = metainfo["structure"]
                    if structure:
                        for seg in structure[:10]:  # Show first 10 segments
                            st.write(f"• **{seg['label']}** at {seg['time']}s")
                    else:
                        st.write("No structure segments detected")
                
                # Generate prompt
                st.subheader("📝 Generated Prompt")
                instruments_str = ", ".join([f"{inst['instrument']} ({float(inst['confidence']) * 100:.1f}%)" for inst in metainfo["instrument"][:2]])
                structure_labels = ", ".join([f"{seg['label']} (at {seg['time']}s)" for seg in metainfo["structure"]])
                prompt = f"I want to create a {genre} music with {instruments_str}. The structure of the music is {structure_labels}. The BPM of the music is {bpm:.0f}."
                st.text_area("Prompt", prompt, height=100)
                
            except Exception as e:
                st.error(f"Error analyzing music: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
else:
    st.info("👆 Please upload an audio file to get started.")
    st.info("The tool best suits short audio files, less than 30 seconds.")

# Example Music Files Section at the bottom
st.divider()
st.subheader("🎵 Try with Example Music Files")
st.markdown("Select one of the example music files below to test the model:")

# Define example music files (you can add more files to this list)
example_files = [
    {
        "name": "Example Music 1",
        "path": "./example music 1/example1.wav",
        "description": "First example music file for testing"
    },
    {
        "name": "Example Music 2", 
        "path": "./example music 1/example2.wav",
        "description": "Second example music file for testing"
    },
    {
        "name": "Example Music 3",
        "path": "./example music 1/example3.wav",
        "description": "Third example music file for testing"
    }
]

# Create columns for the 3 example files
col1, col2, col3 = st.columns(3)

for idx, (col, example) in enumerate(zip([col1, col2, col3], example_files)):
    with col:
        st.markdown(f"**{example['name']}**")
        st.caption(example['description'])
        
        # Check if file exists
        if os.path.exists(example['path']):
            # Display audio player
            with open(example['path'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            
            # Button to analyze this example
            if st.button(f"Analyze {example['name']}", key=f"analyze_example_{idx}"):
                with st.spinner(f"Analyzing {example['name']}... This may take a moment."):
                    try:
                        metainfo = music_prediction(example['path'])
                        
                        # Display results in columns
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.subheader("🎸 Genre")
                            genre = metainfo["genre"]["result"]
                            confidence = metainfo["genre"]["confidence"]
                            st.metric("Genre", genre, f"{float(confidence) * 100:.1f}% confidence")
                            
                            st.subheader("🎹 Instruments")
                            instruments = metainfo["instrument"]
                            for inst in instruments[:5]:  # Show top 5
                                st.write(f"• **{inst['instrument']}** ({float(inst['confidence']) * 100:.1f}%)")
                        
                        with result_col2:
                            st.subheader("📊 BPM")
                            bpm = metainfo["bpm"]["result"]
                            st.metric("Beats Per Minute", f"{bpm:.0f}")
                            
                            st.subheader("🎼 Structure")
                            structure = metainfo["structure"]
                            if structure:
                                for seg in structure[:10]:  # Show first 10 segments
                                    st.write(f"• **{seg['label']}** at {seg['time']}s")
                            else:
                                st.write("No structure segments detected")
                        
                        # Generate prompt
                        st.subheader("📝 Generated Prompt")
                        instruments_str = ", ".join([f"{inst['instrument']} ({float(inst['confidence']) * 100:.1f}%)" for inst in metainfo["instrument"][:2]])
                        structure_labels = ", ".join([f"{seg['label']} (at {seg['time']}s)" for seg in metainfo["structure"]])
                        prompt = f"I want to create a {genre} music with {instruments_str}. The structure of the music is {structure_labels}. The BPM of the music is {bpm:.0f}."
                        st.text_area("Prompt", prompt, height=100, key=f"prompt_{idx}")
                        
                    except Exception as e:
                        st.error(f"Error analyzing music: {str(e)}")
        else:
            st.warning(f"File not found: {example['path']}")
            st.info("Please add your music file to the project directory and update the path in the code.")
