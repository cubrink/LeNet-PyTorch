#%%
import struct
from pathlib import Path
import numpy as np

# %%
class MNIST:
    """
    Provides interface for loading data from MNIST dataset
    (There are million tools to already do this. This is just for practice.)
    """

    # File format constants
    _HEADER_SIZE_LABEL = 8
    _HEADER_SIZE_IMAGE = 16

    # Default filepaths
    _FILEPATH_DATA = Path(__file__).parent.parent / 'data'
    _FILEPATH_TEST_IMAGE = _FILEPATH_DATA / 't10k-images-idx3-ubyte'
    _FILEPATH_TEST_LABEL = _FILEPATH_DATA / 't10k-labels-idx1-ubyte'
    _FILEPATH_TRAIN_IMG = _FILEPATH_DATA / 'train-images-idx3-ubyte'
    _FILEPATH_TRAIN_LABEL = _FILEPATH_DATA / 'train-labels-idx1-ubyte'
    
    
    @classmethod
    def _read_label_header(cls, header: bytes):
        """
        Reads header file for MNIST label data. Returns size data in header file
        Raises ValueError if invalid magic number read from header
        """
        magic_number, size = struct.unpack('>II', header) # '>' -> big endian, 'II' -> data is 2 uint32's
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number in label header: Expected 2049 got {magic_number}/")
        return size

    @classmethod
    def _read_image_header(cls, header: bytes):
        """
        Reads header file for MNIST image data. Returns size data in header file
        Raises ValueError if invalid magic number read from header
        """
        magic_number, size, rows, cols = struct.unpack('>IIII', header) # '>' -> big endian, 'IIII' -> data is 4 uint32's
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number in label header: Expected 2051 got {magic_number}")
        return size, rows, cols

    @classmethod
    def _read_header(cls, data: bytes, header_type: str):
        """
        Reads size data from file header. Raises exceptions if unable to read header.
        Returns size of data according to metadata
        """
        if header_type not in ['label', 'image']:
            raise ValueError(f"Unknown header type {header_type}")

        # Determine header size and reader
        header_size = MNIST._HEADER_SIZE_LABEL
        header_reader = cls._read_label_header
        if header_type == 'image':
            header_size = MNIST._HEADER_SIZE_IMAGE
            header_reader = cls._read_image_header

        if data.size < header_size:
            raise EOFError("End of file reached before header could be read.")

        # Read header, return size of elements
        return header_reader(data[:header_size])

    @classmethod
    def read_labels(cls, path: str):
        """
        Returns label data from MNIST datasets
        """
        label_bytes = np.fromfile(path, dtype=np.uint8)
        size = cls._read_header(data=label_bytes, header_type='label')
        label_data = np.frombuffer(label_bytes[MNIST._HEADER_SIZE_LABEL:], dtype=np.uint8)
        if label_data.size != size:
            raise ValueError(f"Size metadata ({size}) different from file contents ({label_data.size})")
        return label_data

    @classmethod
    def read_images(cls, path: str):
        """
        Returns image data from MNIST datasets
        """
        image_bytes = np.fromfile(path, dtype=np.uint8)
        size, rows, cols = cls._read_header(data=image_bytes, header_type='image')
        image_data = np.frombuffer(image_bytes[MNIST._HEADER_SIZE_IMAGE:], dtype=np.uint8)
        if divmod(image_data.size, size) != (rows*cols, 0):
            raise ValueError(f"Size metadata ({size * rows * cols}) different from file contents ({image_data.size})")
        # Restructure data to [N, height, width, channels]
        image_data = np.reshape(image_data, (size, rows, cols, 1)) # Grayscale, 1 channel
        return image_data

    @classmethod
    def get_data(cls, format_='numpy', device='cpu'):
        """
        Returns MNIST data as x_train, y_train, x_test, y_test

        By default returns data as numpy arrays.
        Accepted formats: numpy, torch
        """
        if format_.lower() not in ['numpy', 'torch']:
            raise ValueError(f"Unknown format {format_}")

        x_train = cls.read_images(cls._FILEPATH_TRAIN_IMG)
        y_train = cls.read_labels(cls._FILEPATH_TRAIN_LABEL)
        x_test = cls.read_images(cls._FILEPATH_TEST_IMAGE)
        y_test = cls.read_labels(cls._FILEPATH_TEST_LABEL)

        if format_ == 'torch':
            import torch
            x_train, y_train, x_test, y_test = map(torch.Tensor, 
                                                   [x_train, y_train, x_test, y_test])
            x_train, y_train, x_test, y_test = map(lambda t: t.to(device), 
                                                   [x_train, y_train, x_test, y_test])
            
        return x_train, y_train, x_test, y_test
