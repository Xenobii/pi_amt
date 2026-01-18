import jams
import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm


# TODO: Reimplement this

def jams_to_midi(jams_path: Path, midi_path: Path):
    jam = jams.load(str(jams_path))

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)'))

    annotations = jam.search(namespace='note_midi')
    if len(annotations) == 0:
        raise ValueError("No note_midi annotations found")
    
    ann = annotations[0]

    for note in ann:
        pitch = int(round(note.value))
        start = float(note.time)
        end = float(note.time + note.duration)
        velocity = 100
        # TODO: CHECK VELOCITY

        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end
            )
        )

    midi.instruments.append(instrument)
    midi.write(midi_path)



def convert_guitarset(jams_dir: str, midi_dir: str):
    jams_dir = Path(jams_dir)
    midi_dir = Path(midi_dir)
    midi_dir.mkdir(exist_ok=True)

    jams_files = list(jams_dir.glob("*.jams"))
    for jams_file in tqdm(jams_files, total=len(jams_files)):
        midi_file = midi_dir / (jams_file.stem + ".mid")
        jams_to_midi(jams_file, midi_file)

    
if __name__ == "__main__":
    convert_guitarset(
        jams_dir="./datasets/guitarset/annotation",
        midi_dir="./datasets/guitarset/annotation_mid"
    )