import math
import logging
import numpy as np
import tensorflow as tf

from numpy import random
from src.reader.reader_handler import FormatReader
from src.common.enumerations import Shuffle, FileAccess


from src.utils.utility import progress, utcnow
import src.storage.wrapper_db as wrapper_db

class DBReader(FormatReader):

    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self._db = wrapper_db.DbWrapper.get_instance()
        self.db_read_total = 0
    
    def _tf_parse_function(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        features = \
            {
                'image': tf.io.FixedLenFeature([], tf.bytes),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)
        # Get the image as raw bytes.
        dimention = int(math.sqrt(self.record_size))
        image_shape = tf.stack([dimention, dimention, 1])
        image_raw = parsed_example['image']
        label = tf.cast(parsed_example['label'], tf.float32)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.io.decode_raw(image_raw, tf.uint8)
        d = image, label
        return d


    def read(self, epoch_number):
        self.db_read_total = 0
        super().read(epoch_number) # make self.__local_file_list 
        # READ -> TFRecordDataset? raw data?
        # print(self._db.dataset_key)
        
        dataset = []
        for key in self._db.dataset_key:
            self.db_read_total += self._db.db_read(f"{key}", dataset)

        dataset = [element for element in dataset if element is not None]

        if None in dataset:
            raise ValueError(f"Error reading from database")

        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shard(num_shards=self.comm_size, index=self.my_rank)
        # dataset = dataset.map(self._tf_parse_function)
        
        if self.sample_shuffle != Shuffle.OFF:
            if self.sample_shuffle == Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        self._dataset = dataset.batch(self.batch_size, drop_remainder=True) 

    
    
    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            total = math.floor(self.num_samples*len(self._file_list)/self.batch_size/self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        for batch in self._dataset:
            yield batch

    def finalize(self):
        pass

        
