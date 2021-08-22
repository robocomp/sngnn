#include <eigen3/Eigen/Dense>

#include <WebotsAPI.h>
#include <webots/Supervisor.hpp>

using namespace std;
using namespace Eigen;
using namespace webots;


WorldEntity::WorldEntity(WebotsAPI *webotsapi)
{
    this->wapi = webotsapi;
    dynamic = true;
    dynamic_translation_checked = false;
    dynamic_rotation_checked = false;
}

void WorldEntity::set_id(int id)
{
    this->id = id;
}

int WorldEntity::get_id()
{
    return this->id;
}

void WorldEntity::set_handle(Node *handle)
{
    this->handle = handle;
}

Node *WorldEntity::get_handle()
{
    return this->handle;
}

void WorldEntity::remove()
{
    this->handle->remove();
}

void WorldEntity::set_world_position(double x, double z)
{
    this->clear_dynamic_check();
    double y = this->handle->getField("translation")->getSFVec3f()[1];
    double coords[3];
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
    this->handle->getField("translation")->setSFVec3f(coords);
    this->update_tr_matrix();
}

void WorldEntity::set_world_orientation(double angle)
{
    this->clear_dynamic_check();
    double rotation[4];
    rotation[0] = 0.;
    rotation[1] = 1.;
    rotation[2] = 0.;
    rotation[3] = angle;
    this->handle->getField("rotation")->setSFRotation(rotation);
    this->update_tr_matrix();
}

void WorldEntity::get_position(double &x_ret, double &z_ret, WorldEntity *relative_to)
{
    if (this->dynamic or not this->dynamic_translation_checked)
    {
        const double *xyz_ptr = this->handle->getPosition();
        this->x = xyz_ptr[0];
        this->z = xyz_ptr[2];
        this->dynamic_translation_checked = true;
    }

    if (relative_to == NULL)
    {
        x_ret = this->x;
        z_ret = this->z;
    }
    else
    {
        // printf("this\n");
        this->get_orientation(this->angle);
        this->update_tr_matrix();

        // printf("robot\n");
        relative_to->get_position(relative_to->x, relative_to->z);
        relative_to->get_orientation(relative_to->angle);
        relative_to->update_tr_matrix();
        // cout << "ROBOT I" << endl;
        // cout << relative_to->tr_matrix << endl;
        // cout << "GOAL" << endl;
        // cout << this->tr_matrix << endl;
        auto step1 = this->tr_matrix * Vector3f(0., 0., 1.);
        // cout << "step1" << step1;
        auto ret = relative_to->tr_matrix.inverse() * step1;
        x_ret = ret[0];
        z_ret = ret[1];
    }
}

void WorldEntity::get_orientation(double &angle_ret, WorldEntity *relative_to)
{
    if (this->dynamic or not this->dynamic_rotation_checked)
    {
        auto rotation = this->handle->getField("rotation")->getSFRotation();
        this->angle = rotation[3]*rotation[1];
        this->dynamic_rotation_checked = true;
    }

    if (relative_to == NULL)
    {
        angle_ret = this->angle;
        // printf("absolute angle %f\n", angle_ret);
        return;
    }
    else
    {
        double angle0;
        relative_to->get_orientation(angle0);
        double diff_ang = this->angle - angle0;
        angle_ret = atan2(sin(diff_ang), cos(diff_ang));
        // printf("relative angle %f\n", angle_ret);
        return;
    }
}

void WorldEntity::update_tr_matrix()
{
    const double x = this->x;
    const double z = this->z;
    const double a = this->angle;
    tr_matrix << +cos(a), +sin(a),   x,
                 -sin(a), +cos(a),   z,
                    0.0,    0.0,   1.0;
}


void WorldEntity::clear_dynamic_check()
{
    this->dynamic_rotation_checked = false;
    this->dynamic_translation_checked = false;
}
