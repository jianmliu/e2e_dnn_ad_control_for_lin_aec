#!/usr/bin/env python
"""
Convert AEC Challenge synthetic dataset to HDF5 format for DNN-controlled AEC training.

This script reads the Microsoft AEC Challenge synthetic dataset and converts it to
the HDF5 format required by the e2e_dnn_ad_control_for_lin_aec project.

Signal mapping:
    farend_speech      -> u_td_tensor (loudspeaker signal)
    nearend_mic_signal -> y_td_tensor (microphone signal)
    echo_signal        -> d_td_tensor (ground-truth echo)
    nearend_speech     -> s_td_tensor (near-end speech, test only)

Author: Generated for AEC Challenge data conversion
"""

import argparse
import os
import csv
import numpy as np
import h5py
import scipy.io.wavfile as wavfile
from tqdm import tqdm


def load_wav_normalized(filepath):
    """Load WAV file and normalize to float32 in range [-1, 1]."""
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    return sr, data


def parse_meta_csv(meta_path):
    """Parse meta.csv and return list of file entries with split info."""
    entries = []
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'fileid': int(row['fileid']),
                'split': row['split'],
                'nearend_scale': float(row['nearend_scale'])
            })
    return entries


def convert_dataset(aec_path, output_dir, max_train_samples=None, max_test_samples=None):
    """
    Convert AEC Challenge dataset to HDF5 format.

    Args:
        aec_path: Path to AEC Challenge synthetic dataset
        output_dir: Output directory for HDF5 files
        max_train_samples: Maximum number of training samples (None for all)
        max_test_samples: Maximum number of test samples (None for all)
    """
    # Parse metadata
    meta_path = os.path.join(aec_path, 'meta.csv')
    entries = parse_meta_csv(meta_path)

    # Separate train and test entries
    train_entries = [e for e in entries if e['split'] == 'train']
    test_entries = [e for e in entries if e['split'] == 'test']

    print(f"Found {len(train_entries)} train samples, {len(test_entries)} test samples")

    # Limit samples if specified
    if max_train_samples is not None:
        train_entries = train_entries[:max_train_samples]
    if max_test_samples is not None:
        test_entries = test_entries[:max_test_samples]

    print(f"Using {len(train_entries)} train samples, {len(test_entries)} test samples")

    # Define file paths
    farend_dir = os.path.join(aec_path, 'farend_speech')
    mic_dir = os.path.join(aec_path, 'nearend_mic_signal')
    echo_dir = os.path.join(aec_path, 'echo_signal')
    nearend_dir = os.path.join(aec_path, 'nearend_speech')

    # Process training data
    print("\nProcessing training data...")
    train_data = process_entries(
        train_entries, farend_dir, mic_dir, echo_dir, nearend_dir,
        include_nearend_speech=False
    )

    # Save training data
    train_output_path = os.path.join(output_dir, 'train_data.h5')
    save_hdf5(train_output_path, train_data, include_nearend_speech=False)
    print(f"Saved training data to {train_output_path}")

    # Process test data
    print("\nProcessing test data...")
    test_data = process_entries(
        test_entries, farend_dir, mic_dir, echo_dir, nearend_dir,
        include_nearend_speech=True
    )

    # Save test data
    test_output_path = os.path.join(output_dir, 'test_data.h5')
    save_hdf5(test_output_path, test_data, include_nearend_speech=True)
    print(f"Saved test data to {test_output_path}")

    return train_output_path, test_output_path


def process_entries(entries, farend_dir, mic_dir, echo_dir, nearend_dir, include_nearend_speech=False):
    """Process dataset entries and load audio data."""
    u_list = []  # farend/loudspeaker
    y_list = []  # microphone
    d_list = []  # echo
    s_list = []  # nearend speech
    sample_rate = None

    for entry in tqdm(entries, desc="Loading audio"):
        fileid = entry['fileid']

        # Load farend speech (loudspeaker signal)
        farend_path = os.path.join(farend_dir, f'farend_speech_fileid_{fileid}.wav')
        sr, farend = load_wav_normalized(farend_path)
        if sample_rate is None:
            sample_rate = sr

        # Load microphone signal
        mic_path = os.path.join(mic_dir, f'nearend_mic_fileid_{fileid}.wav')
        _, mic = load_wav_normalized(mic_path)

        # Load echo signal
        echo_path = os.path.join(echo_dir, f'echo_fileid_{fileid}.wav')
        _, echo = load_wav_normalized(echo_path)

        # Find minimum length to align signals
        min_len = min(len(farend), len(mic), len(echo))

        u_list.append(farend[:min_len])
        y_list.append(mic[:min_len])
        d_list.append(echo[:min_len])

        if include_nearend_speech:
            # Load nearend speech
            nearend_path = os.path.join(nearend_dir, f'nearend_speech_fileid_{fileid}.wav')
            _, nearend = load_wav_normalized(nearend_path)
            # Apply nearend_scale from meta.csv
            nearend_scaled = nearend[:min_len] * entry['nearend_scale']
            s_list.append(nearend_scaled)

    # Find common length across all samples
    common_len = min(len(arr) for arr in u_list)

    # Stack into tensors
    result = {
        'fs': sample_rate,
        'u_td_tensor': np.stack([arr[:common_len] for arr in u_list], axis=0),
        'y_td_tensor': np.stack([arr[:common_len] for arr in y_list], axis=0),
        'd_td_tensor': np.stack([arr[:common_len] for arr in d_list], axis=0),
    }

    if include_nearend_speech:
        result['s_td_tensor'] = np.stack([arr[:common_len] for arr in s_list], axis=0)

    return result


def save_hdf5(output_path, data, include_nearend_speech=False):
    """Save data dictionary to HDF5 file."""
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('fs', data=data['fs'])
        f.create_dataset('u_td_tensor', data=data['u_td_tensor'], dtype='float64')
        f.create_dataset('y_td_tensor', data=data['y_td_tensor'], dtype='float64')
        f.create_dataset('d_td_tensor', data=data['d_td_tensor'], dtype='float64')

        if include_nearend_speech and 's_td_tensor' in data:
            f.create_dataset('s_td_tensor', data=data['s_td_tensor'], dtype='float64')

    # Print summary
    print(f"  Samples: {data['u_td_tensor'].shape[0]}")
    print(f"  Signal length: {data['u_td_tensor'].shape[1]} samples ({data['u_td_tensor'].shape[1]/data['fs']:.2f}s)")
    print(f"  Sample rate: {data['fs']} Hz")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AEC Challenge dataset to HDF5 format'
    )
    parser.add_argument(
        '--aec_path',
        type=str,
        required=True,
        help='Path to AEC Challenge synthetic dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for HDF5 files'
    )
    parser.add_argument(
        '--max_train',
        type=int,
        default=None,
        help='Maximum number of training samples (default: all)'
    )
    parser.add_argument(
        '--max_test',
        type=int,
        default=None,
        help='Maximum number of test samples (default: all)'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.aec_path):
        raise ValueError(f"AEC path does not exist: {args.aec_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run conversion
    convert_dataset(
        args.aec_path,
        args.output_dir,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test
    )

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
