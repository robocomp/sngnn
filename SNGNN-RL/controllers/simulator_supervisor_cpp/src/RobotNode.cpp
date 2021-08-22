#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <RobotNode.h>


RobotNode::RobotNode(const double &x, const double &z, const double &orient, WebotsAPI *wapi) : WorldEntity(wapi)
{
    this->x = x;
    this->z = z;
    this->dynamic = true;

    std::vector<double> position = {x, 0., z};

    std::ifstream ifs("../../protos/Robot2.wbo");
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    // boost::replace_all(content, "SEEDNAME", this->name);
    boost::replace_all(content, "SEEDX", std::to_string(x));
    boost::replace_all(content, "SEEDZ", std::to_string(z));
    boost::replace_all(content, "SEEDANGLE", std::to_string(orient));

    this->wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef("ROBOT");
    //this->handle = this->wapi->rootChildren->getMFNode(-1);

}

void RobotNode::setOrientation(const double &angle)
{
    const double orientation[4] = {0., 1., 0., angle};
    this->handle->getField("rotation")->setSFRotation(orientation);
}

void RobotNode::stop()
{
    Field *controller_arguments = this->handle->getField("controllerArgs");
    controller_arguments->removeMF(0);
    controller_arguments->removeMF(1);
}


void RobotNode::set_actions(double *commands)
{
    std::string data_str = std::to_string(commands[0]) + ","
                            + std::to_string(commands[1]) + ","
                            + std::to_string(commands[2]) + ","
                            + std::to_string(int(this->collision));
    this->handle->getField("customData")->setSFString(data_str);
}


bool RobotNode::check_collision()
{
    bool ret = false;
    std::string cd = this->handle->getField("customData")->getSFString();

    if (cd.length() > 0)
    {
        ret = bool(float(int(cd.back()-'0')));
    }

    this->collision = ret;
    return ret;

}


// void get_model_bounding_box()
// {
//     x, y = this->get_position()
//     return [
//         x + 1 / 2,
//         x - 1 / 2,
//         y + 1 / 2,
//         y - 1 / 2,
//     ]
// }

