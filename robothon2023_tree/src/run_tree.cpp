#include <ros/ros.h>
#include <skills_util/bt_exec.h>
#include <skills_util/log.h>
#include <ros/package.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "run_tree");
    ros::NodeHandle nh("run_tree_server");
    std::string package_path, package_name, tree_name;

    package_name = "robothon2023_tree";
    package_path = ros::package::getPath(package_name);

    std::string path = package_path + "/tree";
    ROS_INFO_STREAM("Path: "<<path);
    ROS_INFO("Run tree start");

    if (!nh.getParam("/tree_name", tree_name))
    {
        ROS_ERROR_STREAM("No param tree_name");
        exit(0);
    }

    ros::ServiceClient bt_exec_clnt = nh.serviceClient<skills_util_msgs::RunTree>("/skills_util/run_tree");
    ROS_YELLOW_STREAM("Waiting for "<<bt_exec_clnt.getService());
    bt_exec_clnt.waitForExistence();
    ROS_YELLOW_STREAM("Connection ok");

    skills_util_msgs::RunTree bt_exec_msg;
    bt_exec_msg.request.folder_path = path;
    bt_exec_msg.request.tree_name    = tree_name;

    if ( !bt_exec_clnt.call(bt_exec_msg) )
    {
        ROS_ERROR_STREAM("Fail to call service: "<<bt_exec_clnt.getService());
        exit(0);
    }
    if ( bt_exec_msg.response.result == 3)
    {
        ROS_ERROR_STREAM("Execution of init is failed");
        exit(0);
    }

    ROS_INFO("Tree test finish");
    exit(0);
}
