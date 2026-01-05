from music_prediction import music_prediction
metainfo = music_prediction("./blues.00000.wav")

# Extract values from metainfo
genre = metainfo["genre"]["result"]
instruments = ", ".join([f"{inst['instrument']} ({float(inst['confidence']) * 100:.1f}%)" for inst in metainfo["instrument"][:2]])
structure_labels = ", ".join([f"{seg['label']} (at {seg['time']}s)" for seg in metainfo["structure"]])
bpm = metainfo["bpm"]["result"]

print("I want to create a " + genre + " music with " + instruments + ". The structure of the music is " + structure_labels + ". The bpm of the music is " + str(bpm) + ".")