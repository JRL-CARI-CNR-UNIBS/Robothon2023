from enum import Enum
from typing import List
import rospy


class Color(Enum):
    """
        Utility Enum class with color constant definition
    """

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class UserMessages(Enum):
    """
        Utility Enum class with color constant definition
    """

    ### Ros Messages ###
    PARAM_NOT_DEFINED_ERROR = Color.RED.value + "Parameter: {} not defined" + Color.END.value
    PARAM_NOT_WELL_DEFINED = Color.RED.value + "Parameter: {} not well defined" + Color.END.value

    SERVICE_FAILED = Color.RED.value + "Service call failed : {}" + Color.END.value
    SERVICE_CALLBACK = Color.GREEN.value + "Service call {} received" + Color.END.value

    ### Generic Messages ###
    INPUT_MESSAGE_WAIT = "Press Enter to continue..."
    SUCCESSFUL = Color.GREEN.value + "Successfully executed" + Color.END.value
    NOT_SUCCESSFUL = "Not Successfully executed"
    READY = Color.GREEN.value + "Ready" + Color.END.value
    CHECK_OK = Color.GREEN.value + "Check performed correctly" + Color.END.value
    IMPOSSIBLE_TO_GO_ON = Color.RED.value + "--- Impossible to continue ---" + Color.END.value

    ### Custom colored messages
    CUSTOM_RED = Color.RED.value + "{}" + Color.END.value
    CUSTOM_YELLOW = Color.YELLOW.value + "{}" + Color.END.value
    CUSTOM_GREEN = Color.GREEN.value + "{}" + Color.END.value
    CUSTOM_CYAN = Color.CYAN.value + "{}" + Color.END.value
    CUSTOM_DARKCYAN = Color.DARKCYAN.value + "{}" + Color.END.value


def check_params_existence(params_name: List[str]):
    all_params_exist = True
    for param_name in params_name:
        if not rospy.has_param(param_name):
            rospy.logerr(UserMessages.PARAM_NOT_DEFINED_ERROR.value.format(param_name))
            all_params_exist = False
    return all_params_exist
