
from midiutil import MIDIFile
from mingus.core import chords

# chord_progression = ["Cmaj7", "Cmaj7", "Fmaj7", "Gdom7"]

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

errors = {
    'notes': 'Bad input, please refer this spec-\n'
}

chord_progression = ["Cmaj7", "Dmaj7", "Fmaj7", "Gdom7"]

def swap_accidentals(note):
    if note == 'Db':
        return 'C#'
    if note == 'D#':
        return 'Eb'
    if note == 'E#':
        return 'F'
    if note == 'Gb':
        return 'F#'
    if note == 'G#':
        return 'Ab'
    if note == 'A#':
        return 'Bb'
    if note == 'B#':
        return 'C'

    return note


def note_to_number(note: str, octave: int) -> int:
    note = swap_accidentals(note)
    assert note in NOTES, errors['notes']
    assert octave in OCTAVES, errors['notes']

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    assert 0 <= note <= 127, errors['notes']

    return note


array_of_notes = []
for chord in chord_progression:
    temp =[]
    temp.extend(chords.from_shorthand(chord))
    array_of_notes.append(temp)

# array_of_notes = NOTES # ตั้งว่าจะเอาโน๊ตอะไรบ้าง
# print(array_of_notes)

array_of_note_numbers = []
for note in array_of_notes:
    OCTAVE = 4 # เปลี่ยน octive ตรงนี้ 
    temp = []
    for i in note:
        temp.append(note_to_number(i, OCTAVE))
    array_of_note_numbers.append(temp)

# print(array_of_note_numbers)

track = 0
channel = 0
time = 0  # In beats
duration = 3  # In beats
tempo = 120  # In BPM
volume = 100  # 0-127, as per the MIDI standard

for i, pitch in enumerate(array_of_note_numbers):
    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, time, tempo)
    for j in pitch:
        MyMIDI.addNote(track, channel, j, time, duration+i, volume)
        # print("ya")
    # print("-------------")
    print(MyMIDI)
    with open("midi"+chord_progression[i]+".mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)
