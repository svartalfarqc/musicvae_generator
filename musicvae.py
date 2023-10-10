"""
MusicVAE

This code generates samples or interpolation based on magenta checkpoints*. Branched from tutorial Colaboratory notebook**.

* Magenta checkpoints:
https://github.com/magenta/magenta-js/blob/master/music/checkpoints/README.md

** Colaboratory notebook (Copyright 2017 Google LLC.):
https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/MusicVAE.ipynb

"""

print('Importing libraries and defining some helper functions...')
import glob
import magenta

from magenta import music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import tkinter as tk

import os
import sys

# from original Google Colab
#import tensorflow.compat.v1 as tf
# workaround
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# Necessary until pyfluidsynth is updated (>1.2.5).
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set base and script directories
BASE_DIR = os.path.expanduser("~")
CHECKPOINTS_DIR = BASE_DIR + "\\magenta_checkpoints\\"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
GENERATED_DIR = SCRIPT_DIR+"\\generated\\"

# Checkpoints and configurations for melody, drums and trio models

melody_model_options = [
    "mel_16bar_hierdec",
    "mel_16bar_flat",
    "mel_2bar_big",
]
melody_config_options = [
    "hierdec-mel_16bar",
    "flat-mel_16bar",
    "cat-mel_2bar_big",
]
drums_model_options = [
    "drums_2bar_small.lokl",
    "drums_2bar_small.hikl",
    "drums_2bar_nade.reduced",
    "drums_2bar_nade.full",
]
drums_config_options = [
    "cat-drums_2bar_small",
    "nade-drums_2bar_reduced",
    "nade-drums_2bar_full",
]
trio_model_options = [
    "trio_16bar_hierdec",
    "trio_16bar_flat",
]
trio_config_options = [
    "hierdec-trio_16bar",
    "flat-trio_16bar",
]

def save(note_sequence, filename):
  """
  Save note sequence to midi file
  """
  mm.sequence_proto_to_midi_file(note_sequence, filename)

def generate_samples(sample_model, config_name, temperature = 0.5, n=4, length=32 ):
  """
  Generate samples
  """

  config = configs.CONFIG_MAP[config_name]
  model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=CHECKPOINTS_DIR + f"{sample_model}.ckpt")
  samples = model.sample(n=n, length=length, temperature=temperature)

  for i, ns in enumerate(samples):
    save(ns, GENERATED_DIR + f"{sample_model}_sample_{i}.mid")

def interpolate(model, start_seq, end_seq, num_steps, max_length=32,
                assert_same_length=True, temperature=0.5,
                individual_duration=4.0):
  """
  Interpolates between a start and end sequence.
  
  """
  note_sequences = model.interpolate(
      start_seq, end_seq,num_steps=num_steps, length=max_length,
      temperature=temperature,
      assert_same_length=assert_same_length)

  print('Starting Interpolation')
  interp_seq = mm.sequences_lib.concatenate_sequences(
      note_sequences, [individual_duration] * len(note_sequences))
  print('Ended Interpolation')

  return interp_seq if num_steps > 3 else note_sequences[num_steps // 2]

def generate_interpolation(interp_model,config_name, path_pattern = SCRIPT_DIR+"\\*.mid", start_beat = 0, end_beat = 1, temperature = 0.5, individual_duration=4.0, num_steps = 26):
  """
  Generate interpolations
  """

  #midi_files = glob.glob(os.path.join(GENERATED_DIR, file_pattern))
  midi_files = glob.glob(path_pattern)

  input_seqs = []

  for midi_file in midi_files:
    note_sequence = mm.midi_file_to_note_sequence(midi_file)
    input_seqs.append(note_sequence)

  interpolate_config = configs.CONFIG_MAP[config_name]
  interpolate_model = TrainedModel(interpolate_config, batch_size=4, checkpoint_dir_or_path=CHECKPOINTS_DIR + f"{interp_model}.ckpt")

  extracted_beats = []
  for ns in input_seqs:
    extracted_beats.extend(interpolate_config.data_converter.from_tensors(
        interpolate_config.data_converter.to_tensors(ns)[1]))

  start_beat = extracted_beats[start_beat]
  end_beat = extracted_beats[end_beat]

  interp = interpolate(interpolate_model, start_beat, end_beat, num_steps=num_steps, temperature=temperature, individual_duration=individual_duration)

  save(interp, GENERATED_DIR + f"{interp_model}_interp.mid")

def execute_task():
    """
    Executes a task within a list of possible tasks
    Code generated with assistance from ChatGPT. This is an exercise in how ChatGPT can be useful for pair programming.
    """

    selected_task = task_selector.get()
    temperature = float(temperature_entry.get())
    sample_length = int(sample_length_entry.get())
    interp_duration = int(interp_duration_entry.get())
    num_steps=sample_length = int(interp_steps_entry.get())
    begin_index= int(interp_begin_index_entry.get())
    end_index= int(interp_end_index_entry.get())

    if selected_task == "Generate Samples (Melody)":
        selected_model = melody_model_selector.get()
        selected_config = melody_config_selector.get()
        generate_samples(selected_model, selected_config,temperature, sample_length)

    elif selected_task == "Generate Interpolation (Melody)":
        selected_model = melody_model_selector.get()
        selected_config = melody_config_selector.get()
        path = os.path.join(GENERATED_DIR, "mel_*.mid")
        generate_interpolation(selected_model, selected_config, path, begin_index, end_index, temperature, interp_duration, num_steps)
        
    elif selected_task == "Generate Samples (Drums)":
        selected_model = drums_model_selector.get()
        selected_config = drums_config_selector.get()
        generate_samples(selected_model, selected_config, temperature)

    elif selected_task == "Generate Interpolation (Drums)":
        selected_model = drums_model_selector.get()
        selected_config = drums_config_selector.get()
        path = os.path.join(GENERATED_DIR, "drums_*.mid")
        generate_interpolation(selected_model, selected_config, path, begin_index, end_index, temperature, interp_duration, num_steps)

    elif selected_task == "Generate Samples (Trio)":
        selected_model = trio_model_selector.get()
        selected_config = trio_config_selector.get()
        generate_samples(selected_model, selected_config, temperature)

    elif selected_task == "Generate Interpolation (Trio)":
        selected_model = trio_model_selector.get()
        selected_config = trio_config_selector.get()
        path = os.path.join(GENERATED_DIR, "trio_*.mid")
        generate_interpolation(selected_model, selected_config, path, begin_index, end_index, temperature, interp_duration, num_steps)

if __name__ == "__main__":

    """
    Main function

    """

    # Create the main window
    window = tk.Tk()
    window.title("Generate samples or interpolation using MusicVAE")

    # Create a label for task selection
    task_label = tk.Label(window, text="Select a Task:")
    task_label.grid(row=0,column=0, padx=10, pady=5)
    #task_label.pack()

    # Create a dropdown menu for task selection
    task_options = [
        "Generate Samples (Melody)",
        "Generate Interpolation (Melody)",
        "Generate Samples (Drums)",
        "Generate Interpolation (Drums)",
        "Generate Samples (Trio)",
        "Generate Interpolation (Trio)",
    ]

    task_selector = tk.StringVar(window)
    task_selector.set(task_options[0])  # Set the default task
    task_menu = tk.OptionMenu(window, task_selector, *task_options)
    task_menu.grid(row=0,column=1, padx=10, pady=5)
    #task_menu.pack()

    # Create a label for model selection
    model_label = tk.Label(window, text="Select a Model:")
    model_label.grid(row=1,column=1,padx=10,pady=5)
    #model_label.pack()
    # Create a dropdown menu for model selection (Melody)

    melody_model_label = tk.Label(window, text="Melody:")
    melody_model_label.grid(row=2,column=0,padx=10, pady=5)
    #melody_model_label.pack()
    melody_model_selector = tk.StringVar(window)
    melody_model_selector.set(melody_model_options[0])  # Set the default model
    melody_model_menu = tk.OptionMenu(window, melody_model_selector, *melody_model_options)
    melody_model_menu.grid(row=2,column=1,padx=10, pady=5)
    #melody_model_menu.pack()

    # Create a dropdown menu for model selection (Drums)
    drums_model_selector = tk.StringVar(window)
    drums_model_selector.set(drums_model_options[0])  # Set the default model
    drums_model_label = tk.Label(window, text="Drums:")
    drums_model_label.grid(row=3,column=0,padx=10, pady=5)
    #drums_model_label.pack()
    drums_model_menu = tk.OptionMenu(window, drums_model_selector, *drums_model_options)
    drums_model_menu.grid(row=3,column=1,padx=10, pady=5)
    #drums_model_menu.pack()

    # Create a dropdown menu for model selection (Trio)
    trio_model_selector = tk.StringVar(window)
    trio_model_selector.set(trio_model_options[0])  # Set the default model
    trio_model_label = tk.Label(window, text="Trio:")
    trio_model_label.grid(row=4,column=0,padx=10, pady=5)
    #trio_model_label.pack()
    trio_model_menu = tk.OptionMenu(window, trio_model_selector, *trio_model_options)
    trio_model_menu.grid(row=4,column=1,padx=10, pady=5)
    #trio_model_menu.pack()

    # Create a label for configuration selection
    config_label = tk.Label(window, text="Select a Configuration:")
    config_label.grid(row=1,column=2,padx=10, pady=5)
    #config_label.pack()

    # Create a dropdown menu for configuration selection (Melody)
    #melody_config_label = tk.Label(window, text="Melody:")
    #melody_config_label.grid(row=6,column=0,padx=10, pady=5)
    #melody_config_label.pack()
    melody_config_selector = tk.StringVar(window)
    melody_config_selector.set(melody_config_options[0])  # Set the default configuration
    melody_config_menu = tk.OptionMenu(window, melody_config_selector, *melody_config_options)
    melody_config_menu.grid(row=2,column=2,padx=10, pady=5)
    #melody_config_menu.pack()

    # Create a dropdown menu for configuration selection (Drums)
    #drums_config_label = tk.Label(window, text="Drums:")
    #drums_config_label.grid(row=7,column=0,padx=10, pady=5)
    #drums_config_label.pack()
    drums_config_selector = tk.StringVar(window)
    drums_config_selector.set(drums_config_options[0])  # Set the default configuration
    drums_config_menu = tk.OptionMenu(window, drums_config_selector, *drums_config_options)
    drums_config_menu.grid(row=3,column=2,padx=10, pady=5)
    #drums_config_menu.pack()

    #trio_config_label = tk.Label(window, text="Trio:")
    #trio_config_label.grid(row=8,column=0,padx=10, pady=5)
    #trio_config_label.pack()
    trio_config_selector = tk.StringVar(window)
    trio_config_selector.set(trio_config_options[0])  # Set the default configuration
    trio_config_menu = tk.OptionMenu(window, trio_config_selector, *trio_config_options)
    trio_config_menu.grid(row=4,column=2,padx=10, pady=5)
    #trio_config_menu.pack()

    # Create entry fields for additional parameters
    temperature_label = tk.Label(window, text="Temperature:")
    temperature_label.grid(row=6, column=0, padx=10, pady=5)
    temperature_entry = tk.Entry(window)
    temperature_entry.insert(0, "0.5")  # Default value
    temperature_entry.grid(row=6, column=1, padx=10, pady=5)

    sample_length_label = tk.Label(window, text="Sample Length:")
    sample_length_label.grid(row=7, column=0, padx=10, pady=5)
    sample_length_entry = tk.Entry(window)
    sample_length_entry.insert(0, "32")  # Default value
    sample_length_entry.grid(row=7, column=1, padx=10, pady=5)

    interp_additional_label = tk.Label(window, text="Interpolation parameters:")
    interp_additional_label.grid(row=8,column=0,padx=10, pady=5)

    interp_duration_label = tk.Label(window, text="Individual Duration:")
    interp_duration_label.grid(row=9, column=0, padx=10, pady=5)
    interp_duration_entry = tk.Entry(window)
    interp_duration_entry.insert(0, "32")  # Default value, 4 is good for 2 bars, 32 for 16 bars
    interp_duration_entry.grid(row=9, column=1, padx=10, pady=5)

    interp_steps_label = tk.Label(window, text="Number of steps:")
    interp_steps_label.grid(row=9, column=2, padx=10, pady=5)
    interp_steps_entry = tk.Entry(window)
    interp_steps_entry.insert(0, "6")  # Default value
    interp_steps_entry.grid(row=9, column=3, padx=10, pady=5)

    interp_begin_index_label = tk.Label(window, text="Begin index:")
    interp_begin_index_label.grid(row=10, column=0, padx=10, pady=5)
    interp_begin_index_entry = tk.Entry(window)
    interp_begin_index_entry.insert(0, "0")  # Default value
    interp_begin_index_entry.grid(row=10, column=1, padx=10, pady=5)

    interp_end_index_label = tk.Label(window, text="End index:")
    interp_end_index_label.grid(row=10, column=2, padx=10, pady=5)
    interp_end_index_entry = tk.Entry(window)
    interp_end_index_entry.insert(0, "1")  # Default value
    interp_end_index_entry.grid(row=10, column=3, padx=10, pady=5)

    # Create a button to execute the selected task
    execute_button = tk.Button(window, text="Execute Task", command=execute_task)
    execute_button.grid(row=11,column=0,padx=10,pady=5)
    #execute_button.pack()

    # Start the GUI main loop
    window.mainloop()
