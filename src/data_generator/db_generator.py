
from src.data_generator.data_generator import DataGenerator
from numpy import random
import tensorflow as tf

from src.utils.utility import progress, utcnow
from shutil import copyfile


import logging

import src.storage.wrapper_db as wrapper_db

"""
Generator for creating data in Key-Value format.
"""

class DBGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self._db = wrapper_db.DbWrapper.get_instance()

    def generate(self):
        print("DBGenerator.generate()")
        """
        Generator for creating data in TFRecord format of 3d dataset.
        TODO: Might be interesting / more realistic to add randomness to the file sizes.
        TODO: Extend this to create accurate records for BERT, which does not use image/label pairs.
        """
        super().generate()
        random.seed(10)
        # This creates a 2D image representing a single record
        record_label = 0
        for i in range(self.my_rank, self.total_files_to_generate, self.comm_size):
            progress(i+1, self.total_files_to_generate, "Generating TFRecord Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            if (self._dimension_stdev>0):
                dim1, dim2 = [max(int(d), 0) for d in random.normal( self._dimension, self._dimension_stdev, 2)]
            else:
                dim1 = dim2 = self._dimension
            record = random.random((dim1, dim2))
            # Open a TFRecordWriter for the output-file.
            for i in range(0, self.num_samples):
                # This creates a 2D image representing a single record
                record = random.random((self._dimension, self._dimension))
                img_bytes = record.tobytes()
                data = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[record_label]))
                }
                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)
                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)
                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                self._db.db_write(int(str(int(out_path_spec.split('/')[-1].replace('part', '').replace('.db', '').replace('of', '').replace('_', ''))) + str(i)), serialized)
                # self._db.db_write(f"{out_path_spec.split('/')[-1].replace('part', '').replace('.db', '').replace('of', '').replace('_', '')}", serialized)

        random.seed()
