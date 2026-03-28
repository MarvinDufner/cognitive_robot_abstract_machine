"""Analysis engine for demonstrating tabletop segmentation using stored data.

This module provides an analysis engine that processes stored camera data to perform
tabletop segmentation. It reads data from MongoDB storage and applies a sequence of
annotators for image preprocessing and object detection.

The pipeline consists of the following steps:

1. Reading stored camera data
2. Image preprocessing
3. Point cloud cropping
4. Plane detection
5. Point cloud cluster extraction

.. note::
    This demo uses the MongoDB storage interface and requires a properly configured
    MongoDB database with stored camera data.
"""
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.descriptors import CrDescriptorFactory

from robokudo.analysis_engine import AnalysisEngineInterface

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisAnnotator,
)
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.pipeline import Pipeline
from robokudo.idioms import pipeline_init


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for demonstrating tabletop segmentation from stored data.

    This class implements a basic pipeline for tabletop segmentation using stored
    camera data. It reads data from MongoDB storage and processes it through a
    sequence of annotators to detect and segment objects on a table surface.

    The pipeline includes:

    * Collection reader for accessing stored data
    * Image preprocessing for data preparation
    * Point cloud cropping to focus on relevant regions
    * Plane detection for table surface identification
    * Point cloud cluster extraction for object segmentation

    .. note::
        The pipeline is configured to use MongoDB storage by default and requires
        proper database configuration.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "demo_from_storage"

    def implementation(self) -> Pipeline:
        """Create a basic pipeline for tabletop segmentation.

        This method constructs the processing pipeline by configuring and connecting
        the necessary annotators in sequence.

        :return: The configured pipeline for tabletop segmentation
        """
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                OutlierRemovalOnObjectHypothesisAnnotator(),
                ClusterPoseBBAnnotator(),
                # SlowAnnotator("SlowAnnotator",sleep_in_s=0),
            ]
        )
        return seq
