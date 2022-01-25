import h5py
import logging

# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, name, reference, count=0):
        self.name = name
        self.count = count
        self.reference = reference


class Serializer:

    def __init__(self):
        self.file = None
        self.datasets = dict()

    def open(self, file_name):

        if self.file:
            logger.info('File ' + self.file.name + ' is currently open - will close it')
            self.close_file()

        logger.info('Open file ' + file_name)
        self.file = h5py.File(file_name, "w")

    def close(self):
        self.compact_data()

        logger.info('Close file ' + self.file.name)
        self.file.close()

    def compact_data(self):
        # Compact datasets, i.e. shrink them to actual size

        for key, dataset in self.datasets.items():
            if dataset.count < dataset.reference.shape[0]:
                logger.info('Compact data for dataset ' + dataset.name + ' from ' + str(
                    dataset.reference.shape[0]) + ' to ' + str(dataset.count))
                dataset.reference.resize(dataset.count, axis=0)

    def append_dataset(self, dataset_name, value, dtype="f8", shape=[1, ], compress=False):
        # print(dataset_name, value)

        # Create dataset if not existing
        if dataset_name not in self.datasets:

            dataset_options = {}
            if compress:
                compression = "gzip"
                compression_opts = 5
                shuffle = True
                dataset_options = {'shuffle': shuffle}
                if compression != 'none':
                    dataset_options["compression"] = compression
                    if compression == "gzip":
                        dataset_options["compression"] = compression_opts

            reference = self.file.require_dataset(dataset_name, [1, ] + shape, dtype=dtype, maxshape=[None, ] + shape,
                                                  **dataset_options)
            self.datasets[dataset_name] = Dataset(dataset_name, reference)

        dataset = self.datasets[dataset_name]
        # Check if dataset has required size, if not extend it
        if dataset.reference.shape[0] < dataset.count + 1:
            dataset.reference.resize(dataset.count + 1000, axis=0)

        # TODO need to add an None check - i.e. for different frequencies
        if value is not None:
            dataset.reference[dataset.count] = value

        dataset.count += 1
