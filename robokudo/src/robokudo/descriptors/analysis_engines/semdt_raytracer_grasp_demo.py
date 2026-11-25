"""
Analysis engine that predicts the grasps of the HSR's gripper on the objects of the
SemDT tabletop the ray tracer renders.
"""

from __future__ import annotations

# %% Imports
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.grasp_pose_annotator import GraspPoseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    WorldDescriptorSource,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.grasp_prediction import GraspGenXGripper
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB

# %% analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_raytracer_grasp_demo"

    def implementation(self) -> Pipeline:
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            RayTracingCameraConfiguration.registry_name,
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_tabletop"
            ),
        )
        plane_descriptor = PlaneAnnotator.Descriptor()
        plane_descriptor.parameters.distance_threshold = 0.01
        grasp_descriptor = GraspPoseAnnotator.Descriptor()
        hsr = HSRB.from_world(
            URDFParser.from_file(file_path=HSRB.get_ros_file_path()).create_world()
        )
        grasp_descriptor.parameters.gripper = GraspGenXGripper.from_two_finger_gripper(
            gripper=hsr.all_arms[0].end_effector
        )

        pipeline = Pipeline("SemDTRayTracerGraspPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_descriptor),
                PointCloudClusterExtractor(),
                GraspPoseAnnotator(descriptor=grasp_descriptor),
            ]
        )
        return pipeline
