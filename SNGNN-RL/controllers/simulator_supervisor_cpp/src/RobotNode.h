#pragma once

#include <webots/Supervisor.hpp>

#include <WorldEntity.h>

using namespace webots;


class RobotNode : public WorldEntity
{
public:

    RobotNode(const double &x, const double &z, const double &angle, WebotsAPI *wapi);
    void stop();
    void set_actions(double *commands);

    void setOrientation(const double &angle);

    // void get_velocity()
    // {
    //     return this->handle->getVelocity();
    // }

    bool check_collision();

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

    double actions[4];
    bool collision;
};

