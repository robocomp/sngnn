#pragma once

#include <eigen3/Eigen/Dense>
#include <webots/Supervisor.hpp>

using namespace Eigen;
using namespace webots;


class WebotsAPI;


class WorldEntity
{
public:
    WorldEntity(WebotsAPI *webotsapi);
    void set_id(int id);
    int get_id();
    void set_handle(Node *handle);
    Node *get_handle();
    void remove();
    void set_world_position(double x, double z);
    void set_world_orientation(double angle);
    void get_position(double &x_ret, double &z_ret, WorldEntity *relative_to=NULL);
    void get_orientation(double &angle_ret, WorldEntity *relative_to=NULL);


// protected:
    double x;
    double z;
    double angle;
    Node *handle;
    int id;
    bool dynamic;
    WebotsAPI *wapi;
    Field *rootChildren;


    void clear_dynamic_check();
protected:
    void update_tr_matrix();
    Matrix3f tr_matrix;
    bool dynamic_rotation_checked;
    bool dynamic_translation_checked;


};
