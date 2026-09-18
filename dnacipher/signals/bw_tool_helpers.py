import numpy as np

class BwtoolBinaryMatrixFile:
    """ Copied from: https://github.com/CRG-Barcelona/bwtool/pull/52
    """
    BWTOOL_MAGIC = 0x90916c6f6f547742
    BWTOOL_FILEFORMAT_VERSION = 1
    BWTOOL_BINARY_MATRIX = 0
    BWTOOL_BINARY_CLUSTER_MATRIX = 1

    def __init__(self, bwtool_binary_matrix_filename):
        self.bwtool_binary_matrix_filename = bwtool_binary_matrix_filename

        self.bwtool_binary_matrix_fh = open(bwtool_binary_matrix_filename, 'r')

        self.bwtool_binary_matrix_initialized = False
        self.text_metadata_initialized = False

        self._bwtool_binary_matrix = None
        self._text_metadata = None

        # Read the header of the bwtool binary matrix file.
        (self.bwtool_magic,
         self.bwtool_fileformat_version,
         self.bwtool_binary_matrix_type,
         self.bwtool_binary_matrix_nbr_rows,
         self.bwtool_binary_matrix_nbr_columns,
         self.bwtool_binary_matrix_float_byte_size,
         self.bwtool_binary_matrix_length_in_bytes,
         self.bwtool_binary_matrix_offset,
         self.text_metadata_offset) = np.fromfile(self.bwtool_binary_matrix_fh,
                                                  dtype=np.uint64,
                                                  count=9)

        # Do some checking of the file format:
        if self.bwtool_magic != BwtoolBinaryMatrixFile.BWTOOL_MAGIC:
            raise ValueError(
                '"{0:s}" is not a bwtool binary matrix file.'.format(
                    self.bwtool_binary_matrix_filename
                )
            )

        if self.bwtool_fileformat_version != BwtoolBinaryMatrixFile.BWTOOL_FILEFORMAT_VERSION:
            raise ValueError(
                '"{0:s}" contains an unsupported binary matrix file version {1:d}.\n'.format(
                    self.bwtool_binary_matrix_filename,
                    self.bwtool_fileformat_version
                ) +
                'bwtool_binary_matrix only supports binary matrix file version {0:d}.'.format(
                    BwtoolBinaryMatrixFile.BWTOOL_FILEFORMAT_VERSION
                )
            )

        # Set the right float size for the bwtool binary matrix.
        if self.bwtool_binary_matrix_float_byte_size == 2:
            self.bwtool_binary_matrix_numpy_float = np.float16
        elif self.bwtool_binary_matrix_float_byte_size == 4:
            self.bwtool_binary_matrix_numpy_float = np.float32
        elif self.bwtool_binary_matrix_float_byte_size == 8:
            self.bwtool_binary_matrix_numpy_float = np.float64
        elif self.bwtool_binary_matrix_float_byte_size == 16:
            self.bwtool_binary_matrix_numpy_float = np.float128
        else:
            raise ValueError(
                'Unsupported float byte size of {0:d} in bwtool binary matrix file "{1:s}".'.format(
                    self.bwtool_binary_matrix_float_byte_size,
                    self.bwtool_binary_matrix_filename
                )
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close file handle.
        self.bwtool_binary_matrix_fh.close()

    def __str__(self):
        return (
            '<class BwtoolBinaryMatrixFile>\n  ' + '\n  '.join(
                [
                    'bwtool_binary_matrix_filename: {0:s}'.format(self.bwtool_binary_matrix_filename),
                    'bwtool_fileformat_version: {0:d}'.format(self.bwtool_fileformat_version),
                    'bwtool_binary_matrix_type: {0:d}'.format(self.bwtool_binary_matrix_type),
                    'bwtool_binary_matrix_type_str: {0:s}'.format(self.bwtool_binary_matrix_type_str),
                    'bwtool_binary_matrix_nbr_rows: {0:d}'.format(self.bwtool_binary_matrix_nbr_rows),
                    'bwtool_binary_matrix_nbr_columns: {0:d}'.format(self.bwtool_binary_matrix_nbr_columns),
                    'bwtool_binary_matrix_float_byte_size: {0:d}'.format(self.bwtool_binary_matrix_float_byte_size),
                    'bwtool_binary_matrix_numpy_float: {0:s}'.format(self.bwtool_binary_matrix_numpy_float),
                    'bwtool_binary_matrix_length_in_bytes: {0:d}'.format(self.bwtool_binary_matrix_length_in_bytes),
                    'bwtool_binary_matrix_offset: {0:d}'.format(self.bwtool_binary_matrix_offset),
                    'text_metadata_offset: {0:d}'.format(self.text_metadata_offset)
                ]) + '\n'
        )

    def is_bwtool_binary_matrix(self):
        return self.bwtool_binary_matrix_type == BwtoolBinaryMatrixFile.BWTOOL_BINARY_MATRIX

    def is_bwtool_binary_cluster_matrix(self):
        return self.bwtool_binary_matrix_type == BwtoolBinaryMatrixFile.BWTOOL_BINARY_CLUSTER_MATRIX

    @property
    def bwtool_binary_matrix(self):
        if self.bwtool_binary_matrix_initialized is False:
            # Seek till the beginning of the bwtool binary matrix.
            self.bwtool_binary_matrix_fh.seek(self.bwtool_binary_matrix_offset, 0)

            # Read the bwtool binary matrix and reshape to the correct dimensions.
            self._bwtool_binary_matrix = np.fromfile(
                self.bwtool_binary_matrix_fh,
                dtype=self.bwtool_binary_matrix_numpy_float,
                count=(self.bwtool_binary_matrix_nbr_rows * self.bwtool_binary_matrix_nbr_columns)
            ).reshape(self.bwtool_binary_matrix_nbr_rows, self.bwtool_binary_matrix_nbr_columns)

            self.bwtool_binary_matrix_initialized = True

        return self._bwtool_binary_matrix

    @property
    def text_metadata(self):
        if self.text_metadata_initialized is False:
            if self.text_metadata_offset == 0:
                # If text metadata offset is set to 0, there is no text metadata.
                self._text_metadata = None
            else:
                # Seek till the beginning of the text metadata.
                self.bwtool_binary_matrix_fh.seek(self.text_metadata_offset, 0)
                # Read the text metadata and split on newlines.
                self._text_metadata = np.fromfile(self.bwtool_binary_matrix_fh,
                                                  dtype='|S1',
                                                  count=-1).tostring().split('\n')[:-1]

            self.text_metadata_initialized = True

        return self._text_metadata

    @property
    def bwtool_binary_matrix_type_str(self):
        if self.bwtool_binary_matrix_type == BwtoolBinaryMatrixFile.BWTOOL_BINARY_MATRIX:
            return 'BWTOOL_BINARY_MATRIX'
        elif self.bwtool_binary_matrix_type == BwtoolBinaryMatrixFile.BWTOOL_BINARY_CLUSTER_MATRIX:
            return 'BWTOOL_BINARY_CLUSTER_MATRIX'
        else:
            return None
