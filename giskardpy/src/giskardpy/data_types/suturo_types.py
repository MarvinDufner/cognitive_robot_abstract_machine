from enum import unique, Enum


# States for the HSRs gripper, is being used only in old_python_interface as of now
@unique
class GripperTypes(Enum):
    OPEN = "open"
    CLOSE = "close"
    NEUTRAL = "neutral"


# List of Methods that need Thresholds for force_monitor
@unique
class ForceTorqueThresholds(Enum):
    GRASP = 1
    PLACE = 2
    DOOR = 3
    DISHDOOR = 4
    POURING = 5  # Pouring (not currently in use)
    SHELF_GRASP = 6
    WIPING = 7
    HRI_GRASP = 8


# List of Objects that need to be differentiated between when placing method is used
@unique
class ObjectTypes(Enum):
    OT_Default = "Default"  # Normal Objects(e.g Milk), includes Cups/Glasses, since planning grabs them from front
    OT_Cutlery = "Cutlery"
    OT_Plate = "Plate"
    OT_Bowl = "Bowl"
    OT_Tray = "Tray"


# List of possible grasping directions
@unique
class GraspTypes(Enum):
    FRONT = "front"
    ABOVE = "above"
    LEFT = "left"
    RIGHT = "right"
    BELOW = "below"


# List of possible poses
@unique
class TakePoseTypes(Enum):
    PARK = "park"
    PARK_LEFT = "park_left"
    PERCEIVE = "perceive"
    ASSISTANCE = "assistance"
    PRE_ALIGN_HEIGHT = "pre_align_height"
    CARRY = "carry"
    TEST = "test"
    PRE_TRAY = "pre_tray"


@unique
class MoveAroundHingeAlign(int, Enum):
    LAST = 0
    ALL = 1
